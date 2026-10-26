# C Programming Deep-Dive — Embedded Interview & MCQ Prep

---

## 1. Pointers, Pointer Arithmetic, Structs, Unions, Bitfields, Function Pointers

### 1.1 Pointers — Core Concepts

A pointer is a variable that stores a memory address.

```c
int x = 10;
int *p = &x;      // p holds address of x
printf("%d", *p);  // dereference: prints 10
*p = 20;            // modifies x through p
```

**Levels of indirection:**
```c
int x = 5;
int *p = &x;
int **pp = &p;      // pointer to pointer
**pp = 100;          // x is now 100
```

**Pointer to const vs const pointer:**
```c
const int *p1;       // pointer to const int — value can't change via p1
int *const p2 = &x;  // const pointer — p2 can't point elsewhere
const int *const p3 = &x; // neither can change
```
Read right-to-left from the `*`: `const int *p1` → "p1 is a pointer to a const int."

### 1.2 Pointer Arithmetic

Pointer arithmetic is scaled by the size of the pointed-to type.

```c
int arr[5] = {10, 20, 30, 40, 50};
int *p = arr;        // decays to &arr[0]
p++;                  // now points to arr[1], address += sizeof(int)
printf("%d", *p);     // 20
printf("%d", *(p+2)); // arr[3] = 40
printf("%ld", (char*)(p+1) - (char*)p); // prints sizeof(int), e.g. 4
```

Key facts:
- `arr[i]` is literally `*(arr + i)` — that's why `i[arr]` also works (weird but valid).
- Subtracting two pointers of the same type gives the number of elements between them, not bytes.
- Pointer arithmetic on `void*` is illegal in standard C (GCC allows it as an extension, treating it like `char*`).
- Incrementing a pointer past the array bounds (beyond one-past-the-end) is undefined behavior.

### 1.3 Structs

```c
struct Sensor {
    uint8_t  id;
    uint16_t value;
    uint32_t timestamp;
};

struct Sensor s = {1, 512, 1000};
struct Sensor *ps = &s;
ps->value = 600;       // same as (*ps).value = 600;
```

**Struct padding/alignment** (very common interview topic):
```c
struct A {
    char  c;   // 1 byte
    int   i;   // 4 bytes
    char  c2;  // 1 byte
};
// sizeof(struct A) is typically 12, not 6, due to alignment padding
```
The compiler inserts padding so each member starts at an address that's a multiple of its own alignment requirement (usually its size, up to the platform word size). Reordering members to go largest-to-smallest often reduces padding:
```c
struct B {
    int   i;   // 4 bytes
    char  c;   // 1 byte
    char  c2;  // 1 byte
    // 2 bytes padding to align struct to 4
};
// sizeof(struct B) is typically 8
```
Use `#pragma pack(1)` or `__attribute__((packed))` to force no padding — critical when a struct maps directly onto a hardware register layout or a wire protocol, but beware of unaligned-access penalties/faults on some MCUs.

### 1.4 Unions

A union stores all members at the **same** memory address; size = size of largest member.

```c
union Data {
    uint32_t whole;
    uint8_t  bytes[4];
};

union Data d;
d.whole = 0x12345678;
printf("%x", d.bytes[0]); // on little-endian MCU: 0x78
```
This is the classic trick used to check endianness or to reinterpret raw register data as bytes.

### 1.5 Bitfields

Bitfields let you pack sub-byte-width members into a struct — commonly used to mirror hardware register layouts described in a datasheet/HRM.

```c
typedef struct {
    uint8_t enable   : 1;
    uint8_t mode     : 2;
    uint8_t reserved : 5;
} ControlReg;

ControlReg reg;
reg.enable = 1;
reg.mode = 3;
```
Caveats interviewers love to ask about:
- Bit ordering (MSB vs LSB first) within a bitfield is **implementation-defined**, not guaranteed by the C standard — risky for portable register-mapping code.
- Padding/alignment of bitfields is also implementation-defined.
- Because of this, many HAL codebases prefer explicit masking/shifting (section 4) over bitfields for register access, reserving bitfields for internal, non-portable state flags.

### 1.6 Function Pointers

```c
void uart_send(uint8_t byte) { /* ... */ }
void spi_send(uint8_t byte)  { /* ... */ }

void (*send_func)(uint8_t) = uart_send;
send_func(0x41);          // calls uart_send
send_func = spi_send;     // now calls spi_send
```

**Why this matters in HAL design:** function pointers are how you build driver "vtables" — a generic HAL API that dispatches to chip-specific implementations at runtime, without `#ifdef` spaghetti.

```c
typedef struct {
    void (*init)(void);
    void (*write)(uint8_t data);
    uint8_t (*read)(void);
} UART_Driver;

UART_Driver uart0 = { .init = uart0_init, .write = uart0_write, .read = uart0_read };
uart0.write(0xFF);
```

Function pointer arrays are also used for interrupt vector tables and command dispatch tables:
```c
typedef void (*ISR)(void);
ISR vector_table[16] = { reset_handler, nmi_handler, hardfault_handler, /* ... */ };
```

---

### MCQs — Section 1

**Q1.** What does `const int *p` mean?
A) p is constant
B) p points to a constant int
C) both p and the int are constant
D) invalid declaration
<details><summary>Answer</summary>B</details>

**Q2.** Given `int arr[10]; int *p = arr;`, what does `*(p+3)` refer to?
A) arr[0] address + 3 bytes
B) arr[3]
C) undefined behavior
D) p + 3 bytes literally added to pointer value
<details><summary>Answer</summary>B — pointer arithmetic scales by sizeof(int).</details>

**Q3.** What is `sizeof(struct A)` given:
```c
struct A { char c; int i; char c2; };
```
(Assume 4-byte int alignment)
A) 6
B) 9
C) 12
D) 8
<details><summary>Answer</summary>C (typically), due to padding after `c` and at the end of the struct.</details>

**Q4.** In a union `union U { int i; char c[4]; }`, after `u.i = 1;` on a little-endian machine, what is `u.c[0]`?
A) 0
B) 1
C) 4
D) undefined
<details><summary>Answer</summary>B — least significant byte stored first in little-endian.</details>

**Q5.** Why are bitfields risky for portable hardware register mapping?
A) They can't represent more than 1 bit
B) Bit ordering and padding are implementation-defined
C) They consume more memory than masks
D) They can't be used in structs
<details><summary>Answer</summary>B</details>

**Q6.** What does `void (*fp)(int)` declare?
A) A function returning a pointer to int
B) A pointer to a function taking an int and returning void
C) An array of function pointers
D) Invalid syntax
<details><summary>Answer</summary>B</details>

**Q7.** Subtracting two `int*` pointers 8 bytes apart in memory (on a 4-byte-int system) yields:
A) 8
B) 2
C) 0
D) Undefined unless same array
<details><summary>Answer</summary>B (and technically well-defined only if both point into, or one-past, the same array)</details>

---

## 2. Memory Layout — Stack, Heap, .data/.bss/.text, Linker Basics

### 2.1 Program Memory Segments

```
High addresses
+-------------------+
|   Stack            |  <- grows downward; local vars, return addrs, saved regs
|        ↓            |
|                     |
|        ↑            |
|   Heap              |  <- grows upward; malloc/free (dynamic memory)
+-------------------+
|   .bss              |  <- uninitialized globals/statics (zero-initialized at startup)
+-------------------+
|   .data             |  <- initialized globals/statics (nonzero values, stored in image)
+-------------------+
|   .text (.rodata)   |  <- code, and read-only constants/string literals
+-------------------+
Low addresses
```

```c
int   g_uninit;             // .bss  (zero by default)
int   g_init = 5;           // .data
const char *msg = "hi";     // "hi" string literal in .rodata; msg pointer itself in .data
static int s_counter;       // .bss (static storage duration, zero-initialized)

void foo(void) {
    int local = 10;         // stack
    int *p = malloc(4);     // p is on stack, the 4 bytes it points to are on heap
}
```

### 2.2 Why This Matters on Embedded Targets

- On a microcontroller, `.data` must be **copied from Flash into RAM** at startup (since RAM is volatile and initial values live in Flash), and `.bss` must be **zeroed** — this is exactly what the startup code (`Reset_Handler` / `crt0`) does before `main()` runs.
- Heap is often avoided or tightly bounded in embedded/real-time firmware because of fragmentation and non-deterministic allocation time — many HAL/platform layers use static or pool allocation instead of `malloc`.
- Stack overflow is a classic embedded bug: deep recursion or large local arrays (e.g., `uint8_t buf[4096]` on the stack) can silently corrupt adjacent memory since MCUs often lack MMU-based guard pages.

### 2.3 Linker Script Basics

A linker script tells the linker where to place each section in the target's actual memory map (Flash vs RAM addresses), since the MCU's memory map is fixed by silicon.

```
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}

SECTIONS
{
  .text : { *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM AT> FLASH   /* loaded in flash, run from RAM */
  .bss  : { *(.bss*) }  > RAM
}
```
Key idea: `.data`'s **load address (LMA)** is in Flash, but its **run address (VMA)** is in RAM — the startup code copies LMA→VMA before `main()`.

### 2.4 Storage Duration & Linkage Quick Reference

| Qualifier | Storage duration | Default linkage | Segment |
|---|---|---|---|
| local var | automatic | none | stack |
| `static` local var | static | none | .data/.bss |
| global var | static | external | .data/.bss |
| `static` global var | static | internal (file-scope only) | .data/.bss |
| `malloc`'d memory | dynamic | n/a | heap |

---

### MCQs — Section 2

**Q1.** Where does an initialized global variable `int x = 5;` live?
A) .bss
B) .data
C) .text
D) Stack
<details><summary>Answer</summary>B</details>

**Q2.** Where is `static int count;` (uninitialized, file scope) stored?
A) Heap
B) Stack
C) .bss
D) .rodata
<details><summary>Answer</summary>C</details>

**Q3.** On an MCU, why must `.data` be copied from Flash to RAM at startup?
A) Flash is faster than RAM
B) RAM is volatile, so initial values must persist in non-volatile Flash and be copied in before use
C) The compiler requires it
D) .data doesn't need copying, only .bss does
<details><summary>Answer</summary>B</details>

**Q4.** What does the linker do with `.bss` at startup on a typical embedded target?
A) Copies it from Flash to RAM
B) Leaves it uninitialized
C) Zero-initializes it in RAM (via startup code, not the linker itself, but scripted by it)
D) Deletes it
<details><summary>Answer</summary>C</details>

**Q5.** Which grows toward lower memory addresses on most architectures?
A) Heap
B) Stack
C) .text
D) .data
<details><summary>Answer</summary>B</details>

**Q6.** Why is heap allocation (`malloc`) often avoided in embedded firmware?
A) malloc doesn't exist in C
B) Fragmentation and non-deterministic timing conflict with real-time requirements
C) Heap memory is read-only
D) It's slower than the stack always fails on MCUs
<details><summary>Answer</summary>B</details>

**Q7.** What does "LMA" vs "VMA" mean in a linker script?
A) Load Memory Address vs Virtual Memory Address (used loosely as run address in embedded contexts)
B) Least/Most significant Address
C) Local/Main Address
D) They are the same thing
<details><summary>Answer</summary>A</details>

---

## 3. `volatile`, `const`, `static` — and Why `volatile` Matters for Hardware Registers

### 3.1 `const`

Tells the compiler the value shouldn't be modified through this identifier — enables compiler warnings/errors and can allow placement in read-only memory (`.rodata`).

```c
const float PI = 3.14159f;
// PI = 3.0; // compile error
```

### 3.2 `static`

Three distinct meanings depending on context — a favorite interview trap:

```c
// 1. Inside a function: retains value between calls (static storage duration)
void counter(void) {
    static int calls = 0;
    calls++;
}

// 2. At file scope: limits linkage to this translation unit (internal linkage)
static int module_private_var;
static void helper_function(void) { }

// 3. Inside a struct/class in C++ (not applicable to plain C structs the same way)
```

### 3.3 `volatile` — Critical for Embedded

`volatile` tells the compiler: *"This value can change outside the normal flow of this code — do not optimize away or cache reads/writes to it."*

**Without volatile — the bug:**
```c
#define STATUS_REG (*(uint32_t *)0x40000000)

void wait_ready(void) {
    while (STATUS_REG == 0) {
        // busy wait
    }
}
```
An optimizing compiler may read `STATUS_REG` into a CPU register **once**, then loop forever checking that cached register value — because from the compiler's point-of-view, nothing in this loop body changes the memory. If hardware changes the actual register value, your code never sees it and hangs forever.

**Fix:**
```c
#define STATUS_REG (*(volatile uint32_t *)0x40000000)
```
Now the compiler re-reads memory on every access.

**Where `volatile` is required:**
1. Memory-mapped hardware registers (status flags, data registers)
2. Global variables modified inside an ISR and read in `main()` (or vice versa)
3. Variables shared across threads/tasks without other synchronization (though for true thread-safety you generally also need atomics or a mutex — `volatile` alone does NOT guarantee atomicity)

```c
volatile uint8_t rx_flag = 0;

void UART_IRQHandler(void) {
    rx_flag = 1;     // set from ISR
}

void main_loop(void) {
    while (!rx_flag) { }   // read in main — must be volatile or compiler may cache it
    rx_flag = 0;
}
```

**Common misconception:** `volatile` is not a substitute for proper synchronization primitives (mutexes, atomic operations) for multi-core race conditions — it only prevents compiler-level caching/reordering of that variable's accesses, not CPU-level reordering or read-modify-write atomicity.

### 3.4 Combining Qualifiers

```c
volatile const uint32_t *STATUS_REG_PTR = (uint32_t *)0x40000000;
// pointer to a volatile const uint32_t:
// - "volatile": hardware can change it, always re-read
// - "const": firmware itself must not write to it (read-only status register)
```

---

### MCQs — Section 3

**Q1.** What problem does `volatile` prevent?
A) Integer overflow
B) The compiler caching a value in a register and never re-reading memory
C) Stack overflow
D) Memory leaks
<details><summary>Answer</summary>B</details>

**Q2.** A variable is written inside an ISR and read in `main()`. It should be declared:
A) const
B) static
C) volatile
D) register
<details><summary>Answer</summary>C</details>

**Q3.** Does `volatile` guarantee atomic access to a variable?
A) Yes, always
B) No — it only prevents compiler caching/optimization, not atomicity
C) Only for 32-bit variables
D) Only inside ISRs
<details><summary>Answer</summary>B</details>

**Q4.** What does `static` mean when applied to a local variable inside a function?
A) The variable is shared across all instances of a struct
B) The variable retains its value between function calls
C) The variable becomes volatile
D) The variable is placed on the heap
<details><summary>Answer</summary>B</details>

**Q5.** What does `static` mean at file (global) scope?
A) Restricts linkage to the current translation unit (internal linkage)
B) Makes the variable read-only
C) Places the variable on the stack
D) Nothing different from a normal global
<details><summary>Answer</summary>A</details>

**Q6.** `volatile uint32_t *const p = (uint32_t*)0x1000;` means:
A) p can point elsewhere but *p can't change
B) p is a constant pointer to a volatile uint32_t — p can't be reassigned, but the pointed-to value may change externally
C) Both p and *p are constant
D) Invalid syntax
<details><summary>Answer</summary>B</details>

**Q7.** Without `volatile` on a hardware status register, what's the most likely failure mode?
A) Compile error
B) Immediate crash
C) Infinite loop / hang because the compiler optimizes away the repeated memory read
D) Incorrect struct padding
<details><summary>Answer</summary>C</details>

---

## 4. Bit Manipulation — Masking, Shifting, Setting/Clearing/Toggling Bits

This is the single most-used C skill in real HAL/register-level work.

### 4.1 The Core Operations

```c
#define BIT(n) (1U << (n))

uint32_t reg = 0;

// SET bit n (OR with mask)
reg |= BIT(3);          // sets bit 3, leaves others unchanged

// CLEAR bit n (AND with inverted mask)
reg &= ~BIT(3);         // clears bit 3, leaves others unchanged

// TOGGLE bit n (XOR with mask)
reg ^= BIT(3);           // flips bit 3

// TEST/CHECK bit n
if (reg & BIT(3)) { /* bit 3 is set */ }

// SET a field of multiple bits (e.g., bits 4-5, a 2-bit mode field)
#define MODE_MASK  (0x3U << 4)
#define MODE_VAL(x) (((x) & 0x3U) << 4)

reg = (reg & ~MODE_MASK) | MODE_VAL(2);   // clear field, then set to value 2
```

### 4.2 Why Read-Modify-Write Matters

`reg |= BIT(3)` is really three operations: **read** reg, **modify** the copy, **write** it back. On embedded systems this has two real hazards:

1. **Non-atomicity:** if an ISR modifies the same register between your read and write, the ISR's change can be lost (classic race condition). Many MCUs solve common cases with dedicated "set" and "clear" registers (e.g., `GPIO_BSRR` on STM32, or Silicon Labs' set/clear/toggle alias registers) so you write `1` to a "set register" to atomically set a bit without a read-modify-write cycle.
```c
// Example: some peripherals expose separate registers so no RMW race exists
GPIO->DOUT_SET = BIT(5);   // atomically sets pin 5, doesn't touch other bits, no read needed
GPIO->DOUT_CLR = BIT(5);   // atomically clears pin 5
```
2. **Write-1-to-clear (W1C) registers:** some status/flag registers clear a bit when you write a `1` to it (not a `0`!) — a very common HRM gotcha. Naively doing `reg &= ~BIT(x)` on such a register can accidentally clear *other* flag bits that happen to read as 1 at that moment.
```c
// WRONG on a W1C register: this clears ALL currently-set flags, not just bit 2
STATUS_REG = STATUS_REG & ~BIT(2);

// RIGHT for W1C semantics: write 1 only to the bit you want to clear
STATUS_REG = BIT(2);
```
Always check the HRM for whether a register is standard read-write, write-1-to-clear, or write-1-to-set before writing masking code.

### 4.3 Extracting a Bitfield

```c
// Extract bits [7:4] (a 4-bit field)
uint32_t value = (reg >> 4) & 0xF;
```

### 4.4 Common Bit Tricks

```c
// Check if a number is a power of two
bool is_pow2(uint32_t n) { return n && !(n & (n - 1)); }

// Count set bits (Brian Kernighan's algorithm)
int popcount(uint32_t n) {
    int count = 0;
    while (n) {
        n &= (n - 1);   // clears the lowest set bit
        count++;
    }
    return count;
}

// Round up to next power of 2 (used in buffer sizing)
uint32_t next_pow2(uint32_t n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// Swap two variables using XOR (no temp — mostly a trivia trick, avoid in real code
// since it fails if the two args alias the same memory)
a ^= b; b ^= a; a ^= b;
```

### 4.5 Signed vs Unsigned Shifts — a Trap

```c
int8_t x = -8;          // 0b11111000
x >> 1;                  // implementation-defined for negative numbers pre-C20-ish in practice;
                          // in GCC/Clang this is arithmetic shift → -4 (sign-extends)

uint8_t u = 0xF8;
u >> 1;                  // logical shift, fills with 0 → 0x7C
```
Right-shifting a **negative signed** value is implementation-defined behavior in C (though virtually all real compilers do arithmetic shift, sign-extending). Left-shifting a signed value such that it overflows, or shifting by ≥ the bit-width of the type, is undefined behavior — both are common review comments in embedded C code.

---

### MCQs — Section 4

**Q1.** Which expression sets bit 5 of `reg` without affecting other bits?
A) `reg = 1 << 5;`
B) `reg |= (1 << 5);`
C) `reg &= (1 << 5);`
D) `reg ^= reg;`
<details><summary>Answer</summary>B</details>

**Q2.** Which expression clears bit 2 of `reg`?
A) `reg |= ~(1 << 2);`
B) `reg &= (1 << 2);`
C) `reg &= ~(1 << 2);`
D) `reg ^= (1 << 2);`
<details><summary>Answer</summary>C</details>

**Q3.** What does `reg ^= (1 << n)` do?
A) Sets bit n
B) Clears bit n
C) Toggles bit n
D) Tests bit n
<details><summary>Answer</summary>C</details>

**Q4.** On a write-1-to-clear (W1C) status register, how do you clear only bit 3 safely?
A) `reg &= ~(1 << 3);`
B) `reg = (1 << 3);`
C) `reg |= (1 << 3);` then `reg = 0;`
D) `reg--;`
<details><summary>Answer</summary>B — write exactly a 1 in that bit position and 0s elsewhere.</details>

**Q5.** Why do some peripherals provide separate "SET" and "CLR" alias registers for GPIO instead of a single read-modify-write DOUT register?
A) To save memory
B) To avoid non-atomic read-modify-write race conditions with ISRs or other bus masters
C) Because C doesn't support bitwise OR
D) They don't actually exist; it's a myth
<details><summary>Answer</summary>B</details>

**Q6.** `(reg >> 4) & 0xF` is used to:
A) Set bits 4-7
B) Extract a 4-bit field starting at bit 4
C) Clear bits 0-3
D) Toggle the entire register
<details><summary>Answer</summary>B</details>

**Q7.** What does `n & (n - 1)` do to the lowest set bit of `n`?
A) Sets it
B) Clears it
C) Toggles it
D) Has no effect
<details><summary>Answer</summary>B — this is the basis of Brian Kernighan's popcount algorithm.</details>

**Q8.** Right-shifting a negative signed integer in standard C is:
A) Always a logical shift (fills with 0)
B) Always an arithmetic shift (sign-extends)
C) Implementation-defined (though arithmetic shift is nearly universal in practice)
D) A compile error
<details><summary>Answer</summary>C</details>

---

## Quick Interview Talking Points (verbal, not just MCQ)

- Be ready to **whiteboard a small register driver**: e.g., "write a function to enable a GPIO pin as output and set it high, given base address and bit position" — combine sections 3 and 4 live.
- Be ready to explain **why `volatile` + `const` together** make sense for a status register pointer.
- Be ready to explain what happens **before `main()` runs** on a microcontroller (stack pointer init, `.data` copy, `.bss` zero, then call `main`) — ties sections 2 and startup code together.
- Practice explaining **struct packing tradeoffs**: packed structs save memory/match wire formats but can cause unaligned access faults or performance penalties on some cores.