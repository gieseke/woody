/*
 * qsort.h
 *
 *  Created on: 12.11.2014
 *      Author: fgieseke
 */

#ifndef INCLUDE_QSORT_H_
#define INCLUDE_QSORT_H_

void woody_qsort(void *base, unsigned num, unsigned width,
		int (*comp)(const void *, const void *, const void *),
		const void* comp_param);

#endif /* INCLUDE_QSORT_H_ */
