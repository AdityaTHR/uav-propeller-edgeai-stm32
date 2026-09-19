CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra -Werror

all: host_parity_test

host_parity_test: host_parity_test.c adaboost_model.c
	$(CC) $(CFLAGS) host_parity_test.c adaboost_model.c -o host_parity_test

test: host_parity_test
	./host_parity_test

clean:
	rm -f host_parity_test
