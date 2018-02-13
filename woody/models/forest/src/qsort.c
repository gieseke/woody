/*
 * qsort.c
 *
 *  Created on: 12.11.2014
 *      Author: fgieseke
 */

#include "include/qsort.h"
//
// qsort.c
//
// Quick sort
//
// Copyright (C) 2002 Michael Ringgaard. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the project nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//

#define INT_CUTOFF 8

static void shortsort(char *lo, char *hi, unsigned width,
		int (*comp)(const void *, const void *, const void*),
		const void *comp_param);

static void woody_swap(char *p, char *q, unsigned int width);

void woody_qsort(void *base, unsigned num, unsigned width,
		int (*comp)(const void *, const void *, const void *),
		const void *comp_param) {

	char *lo, *hi;
	char *mid;
	char *l, *h;
	unsigned size;
	char *lostk[30], *histk[30];
	int stkptr;

	if (num < 2 || width == 0)
		return;

	stkptr = 0;

	lo = base;
	hi = (char *) base + width * (num - 1);

	recurse: size = (hi - lo) / width + 1;

	if (size <= INT_CUTOFF) {
		shortsort(lo, hi, width, comp, comp_param);
	} else {
		mid = lo + (size / 2) * width;
		woody_swap(mid, lo, width);

		l = lo;
		h = hi + width;

		for (;;) {
			do {
				l += width;
			} while (l <= hi && comp(l, lo, comp_param) <= 0);
			do {
				h -= width;
			} while (h > lo && comp(h, lo, comp_param) >= 0);
			if (h < l)
				break;
			woody_swap(l, h, width);
		}

		woody_swap(lo, h, width);

		if (h - 1 - lo >= hi - l) {
			if (lo + width < h) {
				lostk[stkptr] = lo;
				histk[stkptr] = h - width;
				++stkptr;
			}

			if (l < hi) {
				lo = l;
				goto recurse;
			}
		} else {
			if (l < hi) {
				lostk[stkptr] = l;
				histk[stkptr] = hi;
				++stkptr;
			}

			if (lo + width < h) {
				hi = h - width;
				goto recurse;
			}
		}
	}

	--stkptr;
	if (stkptr >= 0) {
		lo = lostk[stkptr];
		hi = histk[stkptr];
		goto recurse;
	}

}

static void shortsort(char *lo, char *hi, unsigned width,
		int (*comp)(const void *, const void *, const void *),
		const void* comp_param) {

	char *p, *max;

	while (hi > lo) {
		max = lo;
		for (p = lo + width; p <= hi; p += width)
			if (comp(p, max, comp_param) > 0)
				max = p;
		woody_swap(max, hi, width);
		hi -= width;
	}

}

static void woody_swap(char *a, char *b, unsigned width) {

	char tmp;

	if (a != b) {
		while (width--) {
			tmp = *a;
			*a++ = *b;
			*b++ = tmp;
		}
	}

}

