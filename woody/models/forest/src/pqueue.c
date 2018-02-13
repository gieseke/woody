// Adapted from http://rosettacode.org/wiki/Priority_queue#C

#include "include/pqueue.h"

/* --------------------------------------------------------------------------------
 * Tests if the queue is empty (first element in array not used to simplify indices)
 * --------------------------------------------------------------------------------
 */
PQUEUE *pqueue_new(int size) {

	if (size < PQUEUE_MIN_SIZE) {
		size = PQUEUE_MIN_SIZE;
	}

	// allocate space for priority queue
	PQUEUE *q = (PQUEUE*) malloc(sizeof(PQUEUE));

	// allocate space for size queue items
	q->buf = (PQUEUE_ITEM*) malloc(size * sizeof(PQUEUE_ITEM));

	// set size and number of elements (first element is not used)
	q->alloc = size;
	q->n = 1;

	return q;
}

/* --------------------------------------------------------------------------------
 * Tests if the queue is empty
 * --------------------------------------------------------------------------------
 */
inline int pqueue_is_empty(PQUEUE *q) {

	if (q->n == 1) {
		return 1;
	} else {
		return 0;
	}

}

/* --------------------------------------------------------------------------------
 * Pushes "data" with priority "pri"
 * --------------------------------------------------------------------------------
 */
void pqueue_push(PQUEUE *q, void *data, int pri) {

	// pointer for queue item
	PQUEUE_ITEM *b;
	int n, m;

	// allocate more memory if needed
	if (q->n >= q->alloc) {
		q->alloc *= 2;
		b = q->buf = (PQUEUE_ITEM*) realloc(q->buf,
				sizeof(PQUEUE_ITEM) * q->alloc);
	} else {
		b = q->buf;
	}

	// append at end and perform an up-heap operation
	// (move up in case parent has a larger priority)
	n = q->n++;
	while ((m = n / 2) && pri < b[m].pri) {
		b[n] = b[m];
		n = m;
	}

	b[n].data = data;
	b[n].pri = pri;

}

/* --------------------------------------------------------------------------------
 * Removes top item (or returns 0 if queue is empty); *pri can be NULL.
 * --------------------------------------------------------------------------------
 */
void *pqueue_pop(PQUEUE *q, int *pri) {

	void *out;
	if (q->n == 1) {
		return 0;
	}

	PQUEUE_ITEM *b = q->buf;

	// get item from the root and store priority in *pri if pri!=NULL
	out = b[1].data;
	if (pri) {
		*pri = b[1].pri;
	}

	// reduce size by one
	--q->n;

	int n = 1, m;
	while ((m = n * 2) < q->n) {

		if (m + 1 < q->n && b[m].pri > b[m + 1].pri) {
			m++;
		}

		if (b[q->n].pri <= b[m].pri) {
			break;
		}

		b[n] = b[m];
		n = m;
	}
	b[n] = b[q->n];

	// reduce size if needed
	if (q->n < q->alloc / 2 && q->n >= PQUEUE_MIN_SIZE) {
		q->buf = (PQUEUE_ITEM*) realloc(q->buf, (q->alloc /= 2) * sizeof(b[0]));
	}

	// return data
	return out;

}

/* --------------------------------------------------------------------------------
 * Returns the top of the queue
 * --------------------------------------------------------------------------------
 */
inline void* pqueue_top(PQUEUE *q, int *pri) {
	if (q->n == 1) {
		return NULL;
	}
	if (pri) {
		*pri = q->buf[1].pri;
	}
	return q->buf[1].data;
}

/* --------------------------------------------------------------------------------
 * Combines/merges two queues
 * --------------------------------------------------------------------------------
 */
void pqueue_combine(PQUEUE *q1, PQUEUE *q2) {
	int i;
	PQUEUE_ITEM *e = q2->buf + 1;

	for (i = q2->n - 1; i >= 1; i--, e++) {
		pqueue_push(q1, e->data, e->pri);
	}

	pqueue_purge(q2);

}

/*int main() {
 int i, p;
 char *c, *tasks[] = { "Clear drains", "Feed cat", "Make tea", "Solve RC tasks", "Tax return" };
 int pri[] = { 3, 4, 5, 1, 2 };

 //make two queues
 PQUEUE *q = pqueue_new(0);
 PQUEUE *q2 = pqueue_new(0);

 //push all 5 tasks into q
 for (i = 0; i < 5; i++)
 pqueue_push(q, tasks[i], pri[i]);

 //pop them and print one by one
 while ((c = pqueue_pop(q, &p)))
 printf("%d: %s\n", p, c);

 //put a million random tasks in each queue
 for (i = 0; i < 1 << 20; i++) {
 p = rand() / ( RAND_MAX / 5);
 pqueue_push(q, tasks[p], pri[p]);

 p = rand() / ( RAND_MAX / 5);
 pqueue_push(q2, tasks[p], pri[p]);
 }

 printf("\nq has %d items, q2 has %d items\n", pqueue_size(q), pqueue_size(q2));

 // merge q2 into q; q2 is empty
 pqueue_combine(q, q2);
 printf("After merge, q has %d items, q2 has %d items\n", pqueue_size(q),
 pqueue_size(q2));

 // pop q until it's empty
 for (i = 0; (c = pqueue_pop(q, 0)); i++)
 ;
 printf("Popped %d items out of q\n", i);

 return 0;
 }*/
