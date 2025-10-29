#include "distance_kernels.h"
DistanceType parse_distance_type(const std::string& arg) {
    if (arg == "l2") return L2;
    if (arg == "l1") return L1;
    if (arg == "cosine") return COSINE;
    throw std::runtime_error("Unknown distance type");
}

