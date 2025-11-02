#include <stdexcept>
#include "distance_kernels.h"
distanceType parse_distance_type(const std::string& arg) {
    if (arg == "L2") return L2;
    if (arg == "L1") return L1;
    if (arg == "Cosine") return COSINE;
    if (arg == "LINF") return LINF;
    throw std::runtime_error("Unknown distance type, must be in {L2, L1, Cosine, LINF}");
}

const char* distanceTypeToString(distanceType type){
	switch (type) {
		case L1: return "L1";
	    case L2: return "L2";
		case COSINE: return "Cosine";
		case LINF: return "LINF";
		default: return "UNKNOWN";
	}
}

