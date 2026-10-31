import json
import unittest
from collections import defaultdict
from sem_yolos.common import ROOT


class CvFoldTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT/'data/splits/split_v2.json').read_text(encoding='utf-8'))
        self.folds = [json.loads((ROOT/f'data/splits/cv5/fold{k}.json').read_text(encoding='utf-8')) for k in range(5)]

    def test_test_split_is_unchanged_in_every_fold(self):
        test = {n for n, s in self.base['assignments'].items() if s == 'test'}
        self.assertEqual(len(test), 13)
        for fold in self.folds:
            self.assertEqual({n for n, s in fold['assignments'].items() if s == 'test'}, test)

    def test_each_image_is_held_out_exactly_once(self):
        held = defaultdict(int)
        for fold in self.folds:
            self.assertEqual(set(fold['assignments']), set(self.base['assignments']))
            for name, split in fold['assignments'].items():
                held[name] += split == 'val'
        for name, split in self.base['assignments'].items():
            self.assertEqual(held[name], 0 if split == 'test' else 1, name)

    def test_groups_never_cross_train_and_held_out(self):
        for fold in self.folds:
            for members in fold['groups'].values():
                self.assertEqual(len({fold['assignments'][m] for m in members}), 1, members)

    def test_gap_defects_sit_in_different_folds(self):
        names = {'fig2': [], 'fig29': [], 'fig14': [], 'fig26': []}
        for k, fold in enumerate(self.folds):
            for name, split in fold['assignments'].items():
                short = name.rsplit('-', 1)[1].removesuffix('.jpg')
                if split == 'val' and short in names:
                    names[short].append(k)
        self.assertEqual(len({v[0] for v in names.values()}), 4, names)


if __name__ == '__main__':
    unittest.main()
