"""Read-only integrity checks for generated offline diagnostic reports."""
import argparse
import csv
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
from core import OUTPUT,read,resolve


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])
        if tag=='img' and 'src' in attrs:self.links.append(attrs['src'])


def verify(combined):
    combined=Path(combined).resolve();pending=[combined/'index.html'];seen=set();links=0
    while pending:
        html=pending.pop()
        if html in seen:continue
        seen.add(html);parser=Links();parser.feed(html.read_text(encoding='utf-8'))
        for link in parser.links:
            parsed=urlsplit(link)
            if parsed.scheme or parsed.netloc:raise ValueError(f'External dependency in offline report: {link}')
            target=(html.parent/unquote(parsed.path)).resolve()
            if not target.is_relative_to(OUTPUT.resolve()):raise ValueError(f'Link outside diagnostic outputs: {target}')
            if not target.is_file():raise ValueError(f'Missing report resource: {target}')
            links+=1
            if target.suffix=='.html':pending.append(target)
    with (combined/'manual_review.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    keys={(r['model'],r['image_id'],r['annotation_id']) for r in rows}
    if len(keys)!=len(rows):raise ValueError('Duplicate manual review keys')
    summaries=read(combined/'summary.json')
    for model,summary in summaries.items():
        subset=[r for r in rows if r['model']==model]
        if len(subset)!=summary['gap_detection']['support']:raise ValueError('Review row count does not match gap support')
        if summary['missing_evidence']:
            for row in subset:
                if row['raw_candidate_evidence']!='unknown_not_captured':raise ValueError('Missing raw evidence not labeled unknown')
                for prefix in ('A','B','C50','C25'):
                    if row[f'{prefix}_iou'] not in ('','None'):raise ValueError('Invented IoU for unavailable candidate')
    return dict(html_pages_checked=len(seen),local_links_checked=links,review_rows=len(rows),
                models=list(summaries),status='passed',note='Read-only HTML/resource/CSV validation; not a model accuracy check.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combined',required=True,type=Path)
    args=p.parse_args();print(json.dumps(verify(resolve(args.combined)),indent=2))
