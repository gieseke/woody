/*
 * heap.h
 *
 *  Created on: 21.10.2014
 *      Author: fgieseke
 */

#ifndef COMMON_INCLUDE_PQUEUE_H_
#define COMMON_INCLUDE_PQUEUE_H_

#include <stdio.h>
#include <stdlib.h>

#define PQUEUE_MIN_SIZE 64

typedef struct {
	void * data;
	int pri;
} PQUEUE_ITEM;

typedef struct {
	PQUEUE_ITEM *buf;
	int n;
	int alloc;
} PQUEUE;

// macros
#define pqueue_purge(q) (q)->n = 1
#define pqueue_size(q) ((q)->n - 1)

/* --------------------------------------------------------------------------------
 * Instantiates a new queue
 * --------------------------------------------------------------------------------
 */
PQUEUE *pqueue_new(int size);

/* --------------------------------------------------------------------------------
 * Tests if the queue is empty
 * --------------------------------------------------------------------------------
 */
inline int pqueue_is_empty(PQUEUE *q);

/* --------------------------------------------------------------------------------
 * Pushes "data" with priority "pri"
 * --------------------------------------------------------------------------------
 */
void pqueue_push(PQUEUE *q, void *data, int pri);

/* --------------------------------------------------------------------------------
 * Removes top item (or returns 0 if queue is empty); *pri can be NULL.
 * --------------------------------------------------------------------------------
 */
void *pqueue_pop(PQUEUE *q, int *pri);

/* --------------------------------------------------------------------------------
 * Returns the top of the queue
 * --------------------------------------------------------------------------------
 */
inline void *pqueue_top(PQUEUE *q, int *pri);

/* --------------------------------------------------------------------------------
 * Combines/merges two queues
 * --------------------------------------------------------------------------------
 */
void pqueue_combine(PQUEUE *q1, PQUEUE *q2);

#endif /* COMMON_INCLUDE_PQUEUE_H_ */
