#pragma once

#define i_j (uint64_t)i << 32 | j
#define i_l (uint64_t)i << 32 | l
#define k_j (uint64_t)k << 32 | j
#define k_l (uint64_t)k << 32 | l
#define i_32 (uint32_t) (ij >> 32)
#define j_32 (uint32_t) ij

#define _theta .x
#define _phi .y
#define thphi_theta thphi.x
#define thphi_phi thphi.y