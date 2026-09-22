#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void read_input() {
    char buf[64];
    int n = 128;
    if (n > 64) {
        memset(buf, 0, n);  // buffer overflow
    }
}

void use_after_free_example() {
    char *ptr = (char *)malloc(64);
    // ... use ptr ...
    free(ptr);
    // ... other work ...
    ptr[0] = 'A';  // use-after-free
}

void unchecked_read() {
    char buf2[64];
    read(0, buf2, 128);  // buffer overflow via read
}

int main() {
    read_input();
    use_after_free_example();
    unchecked_read();
    return 0;
}
