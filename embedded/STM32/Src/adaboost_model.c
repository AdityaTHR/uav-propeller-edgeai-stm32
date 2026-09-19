#include "adaboost_model.h"

static int tree_0(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 274.037964f) {
            return 4;
        } else {
            return 3;
        }
    } else {
        if (x[2] <= 0.231092483f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_1(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[10] <= 0.210430548f) {
            return 0;
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}

static int tree_2(const float *x) {
    if (x[15] <= 281.945129f) {
        if (x[8] <= -0.0181264505f) {
            return 2;
        } else {
            return 0;
        }
    } else {
        if (x[8] <= 0.0215911083f) {
            return 3;
        } else {
            return 1;
        }
    }
}

static int tree_3(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 247.056f) {
            return 4;
        } else {
            return 3;
        }
    } else {
        return 1;
    }
}

static int tree_4(const float *x) {
    if (x[11] <= 1.10459447f) {
        if (x[15] <= 213.237076f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[8] <= -0.0144473091f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_5(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[9] <= 0.181868866f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[4] <= 1.99647999f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_6(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 294.699829f) {
            return 0;
        } else {
            return 3;
        }
    } else {
        if (x[10] <= 0.171218306f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_7(const float *x) {
    if (x[15] <= 223.58432f) {
        if (x[11] <= 1.19381201f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[8] <= -0.0162424557f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_8(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[5] <= -0.049692139f) {
            return 0;
        } else {
            return 4;
        }
    } else {
        if (x[13] <= -0.332323045f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_9(const float *x) {
    if (x[13] <= -0.210644305f) {
        if (x[10] <= 0.196156353f) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (x[15] <= 236.20845f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_10(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[9] <= 0.175535142f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}

static int tree_11(const float *x) {
    if (x[15] <= 218.949097f) {
        if (x[15] <= 204.378174f) {
            return 4;
        } else {
            return 4;
        }
    } else {
        if (x[0] <= 0.041153565f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_12(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[9] <= 0.181868866f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[1] <= 0.227683514f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_13(const float *x) {
    if (x[15] <= 226.68927f) {
        if (x[10] <= 0.239327431f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[8] <= -0.0130708041f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_14(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[10] <= 0.224578291f) {
            return 0;
        } else {
            return 3;
        }
    } else {
        if (x[1] <= 0.227683514f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_15(const float *x) {
    if (x[7] <= 176.156357f) {
        if (x[11] <= 1.07060742f) {
            return 4;
        } else {
            return 3;
        }
    } else {
        if (x[5] <= -0.0795887113f) {
            return 0;
        } else {
            return 2;
        }
    }
}

static int tree_16(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[10] <= 0.160331622f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        return 1;
    }
}

static int tree_17(const float *x) {
    if (x[15] <= 204.378174f) {
        if (x[13] <= -0.169231206f) {
            return 2;
        } else {
            return 4;
        }
    } else {
        if (x[15] <= 295.218384f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_18(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[10] <= 0.194892585f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[0] <= -0.00927558076f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_19(const float *x) {
    if (x[15] <= 245.319016f) {
        if (x[15] <= 202.437531f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[10] <= 0.146951616f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_20(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 229.973694f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[13] <= -0.332323045f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_21(const float *x) {
    if (x[10] <= 0.224876881f) {
        if (x[9] <= 0.168801159f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[0] <= 0.0389103889f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_22(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[8] <= -0.0173497647f) {
            return 2;
        } else {
            return 0;
        }
    } else {
        return 1;
    }
}

static int tree_23(const float *x) {
    if (x[15] <= 236.941895f) {
        if (x[10] <= 0.250559717f) {
            return 4;
        } else {
            return 3;
        }
    } else {
        if (x[8] <= -0.0181144886f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_24(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 219.671448f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        return 1;
    }
}

static int tree_25(const float *x) {
    if (x[9] <= 0.181868866f) {
        if (x[7] <= 183.907867f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[15] <= 219.671448f) {
            return 0;
        } else {
            return 2;
        }
    }
}

static int tree_26(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 294.699829f) {
            return 0;
        } else {
            return 3;
        }
    } else {
        if (x[2] <= 0.231092483f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_27(const float *x) {
    if (x[0] <= 0.0403437354f) {
        if (x[8] <= 0.0102188084f) {
            return 4;
        } else {
            return 1;
        }
    } else {
        if (x[15] <= 219.535004f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_28(const float *x) {
    if (x[8] <= -0.0130553599f) {
        if (x[13] <= -0.175936371f) {
            return 2;
        } else {
            return 2;
        }
    } else {
        if (x[8] <= 0.0110804606f) {
            return 3;
        } else {
            return 1;
        }
    }
}

static int tree_29(const float *x) {
    if (x[15] <= 240.89917f) {
        if (x[13] <= -0.193214625f) {
            return 2;
        } else {
            return 4;
        }
    } else {
        if (x[11] <= 0.743477523f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_30(const float *x) {
    if (x[11] <= 1.30851948f) {
        if (x[3] <= 1.5719285f) {
            return 2;
        } else {
            return 0;
        }
    } else {
        if (x[8] <= 0.0110804606f) {
            return 2;
        } else {
            return 1;
        }
    }
}

static int tree_31(const float *x) {
    if (x[3] <= 1.02387702f) {
        if (x[4] <= 2.51336288f) {
            return 2;
        } else {
            return 3;
        }
    } else {
        if (x[15] <= 291.467834f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_32(const float *x) {
    if (x[10] <= 0.209172189f) {
        if (x[15] <= 241.490829f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[15] <= 207.725616f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_33(const float *x) {
    if (x[11] <= 0.926160455f) {
        if (x[7] <= 183.907867f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[8] <= 0.0110804606f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static int tree_34(const float *x) {
    if (x[0] <= 0.0406437516f) {
        if (x[10] <= 0.213108718f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[11] <= 1.07060742f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_35(const float *x) {
    if (x[15] <= 226.68927f) {
        if (x[13] <= 0.00136513566f) {
            return 0;
        } else {
            return 4;
        }
    } else {
        if (x[8] <= -0.0163046606f) {
            return 2;
        } else {
            return 3;
        }
    }
}

static int tree_36(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[11] <= 1.30851948f) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (x[14] <= -1.09546399f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_37(const float *x) {
    if (x[9] <= 0.159133226f) {
        if (x[7] <= 183.907867f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[7] <= 176.156357f) {
            return 4;
        } else {
            return 2;
        }
    }
}

static int tree_38(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[5] <= -0.0583357401f) {
            return 0;
        } else {
            return 2;
        }
    } else {
        if (x[12] <= 2.14737415f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_39(const float *x) {
    if (x[9] <= 0.20528543f) {
        if (x[10] <= 0.148276433f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[15] <= 219.641632f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_40(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 291.467834f) {
            return 2;
        } else {
            return 3;
        }
    } else {
        return 1;
    }
}

static int tree_41(const float *x) {
    if (x[9] <= 0.197242618f) {
        if (x[4] <= 2.61679363f) {
            return 0;
        } else {
            return 4;
        }
    } else {
        if (x[15] <= 219.169891f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_42(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[15] <= 227.619171f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[14] <= -1.09546399f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_43(const float *x) {
    if (x[9] <= 0.181868866f) {
        if (x[15] <= 296.847046f) {
            return 4;
        } else {
            return 3;
        }
    } else {
        if (x[1] <= 0.280651808f) {
            return 3;
        } else {
            return 0;
        }
    }
}

static int tree_44(const float *x) {
    if (x[8] <= 0.0110804606f) {
        if (x[13] <= -0.173852637f) {
            return 2;
        } else {
            return 0;
        }
    } else {
        if (x[1] <= 0.227683514f) {
            return 1;
        } else {
            return 1;
        }
    }
}

static int tree_45(const float *x) {
    if (x[15] <= 246.920181f) {
        if (x[11] <= 1.23629653f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[10] <= 0.16027528f) {
            return 4;
        } else {
            return 3;
        }
    }
}

static int tree_46(const float *x) {
    if (x[8] <= -0.0130708041f) {
        if (x[7] <= 170.193146f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[8] <= 0.0110804606f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static int tree_47(const float *x) {
    if (x[15] <= 216.160492f) {
        if (x[13] <= -0.27009505f) {
            return 2;
        } else {
            return 4;
        }
    } else {
        if (x[9] <= 0.22351326f) {
            return 0;
        } else {
            return 3;
        }
    }
}

static int tree_48(const float *x) {
    if (x[8] <= -0.0103985798f) {
        if (x[9] <= 0.159133226f) {
            return 4;
        } else {
            return 2;
        }
    } else {
        if (x[8] <= 0.0110804606f) {
            return 3;
        } else {
            return 1;
        }
    }
}

static int tree_49(const float *x) {
    if (x[15] <= 240.009552f) {
        if (x[7] <= 180.904892f) {
            return 4;
        } else {
            return 0;
        }
    } else {
        if (x[3] <= 1.59741855f) {
            return 3;
        } else {
            return 0;
        }
    }
}

static const float TREE_WEIGHTS[UAV_TREE_COUNT] = {0.854366958f, 0.97969979f, 0.8297472f, 0.903445482f, 0.916993856f, 0.75605011f, 0.690455139f, 0.891822696f, 0.7294209f, 0.696121752f, 0.606709599f, 0.710470676f, 0.591507852f, 0.870124817f, 0.665131211f, 0.849161029f, 0.56917882f, 0.72691083f, 0.578167856f, 0.798415482f, 0.629993737f, 0.833005369f, 0.607127547f, 0.730499387f, 0.481176049f, 0.692217886f, 0.490677863f, 0.621075034f, 0.608757079f, 0.634987473f, 0.669760048f, 0.606176972f, 0.660522997f, 0.607221842f, 0.676271856f, 0.727401197f, 0.568357885f, 0.632650256f, 0.580814958f, 0.680574059f, 0.441403359f, 0.575268805f, 0.535379171f, 0.681613326f, 0.516787708f, 0.680783629f, 0.704395831f, 0.62270838f, 0.630631864f, 0.667796135f};

int adaboost_predict(const float x[UAV_FEATURE_COUNT]) {
    float votes[UAV_CLASS_COUNT] = {0};
    votes[tree_0(x)] += TREE_WEIGHTS[0];
    votes[tree_1(x)] += TREE_WEIGHTS[1];
    votes[tree_2(x)] += TREE_WEIGHTS[2];
    votes[tree_3(x)] += TREE_WEIGHTS[3];
    votes[tree_4(x)] += TREE_WEIGHTS[4];
    votes[tree_5(x)] += TREE_WEIGHTS[5];
    votes[tree_6(x)] += TREE_WEIGHTS[6];
    votes[tree_7(x)] += TREE_WEIGHTS[7];
    votes[tree_8(x)] += TREE_WEIGHTS[8];
    votes[tree_9(x)] += TREE_WEIGHTS[9];
    votes[tree_10(x)] += TREE_WEIGHTS[10];
    votes[tree_11(x)] += TREE_WEIGHTS[11];
    votes[tree_12(x)] += TREE_WEIGHTS[12];
    votes[tree_13(x)] += TREE_WEIGHTS[13];
    votes[tree_14(x)] += TREE_WEIGHTS[14];
    votes[tree_15(x)] += TREE_WEIGHTS[15];
    votes[tree_16(x)] += TREE_WEIGHTS[16];
    votes[tree_17(x)] += TREE_WEIGHTS[17];
    votes[tree_18(x)] += TREE_WEIGHTS[18];
    votes[tree_19(x)] += TREE_WEIGHTS[19];
    votes[tree_20(x)] += TREE_WEIGHTS[20];
    votes[tree_21(x)] += TREE_WEIGHTS[21];
    votes[tree_22(x)] += TREE_WEIGHTS[22];
    votes[tree_23(x)] += TREE_WEIGHTS[23];
    votes[tree_24(x)] += TREE_WEIGHTS[24];
    votes[tree_25(x)] += TREE_WEIGHTS[25];
    votes[tree_26(x)] += TREE_WEIGHTS[26];
    votes[tree_27(x)] += TREE_WEIGHTS[27];
    votes[tree_28(x)] += TREE_WEIGHTS[28];
    votes[tree_29(x)] += TREE_WEIGHTS[29];
    votes[tree_30(x)] += TREE_WEIGHTS[30];
    votes[tree_31(x)] += TREE_WEIGHTS[31];
    votes[tree_32(x)] += TREE_WEIGHTS[32];
    votes[tree_33(x)] += TREE_WEIGHTS[33];
    votes[tree_34(x)] += TREE_WEIGHTS[34];
    votes[tree_35(x)] += TREE_WEIGHTS[35];
    votes[tree_36(x)] += TREE_WEIGHTS[36];
    votes[tree_37(x)] += TREE_WEIGHTS[37];
    votes[tree_38(x)] += TREE_WEIGHTS[38];
    votes[tree_39(x)] += TREE_WEIGHTS[39];
    votes[tree_40(x)] += TREE_WEIGHTS[40];
    votes[tree_41(x)] += TREE_WEIGHTS[41];
    votes[tree_42(x)] += TREE_WEIGHTS[42];
    votes[tree_43(x)] += TREE_WEIGHTS[43];
    votes[tree_44(x)] += TREE_WEIGHTS[44];
    votes[tree_45(x)] += TREE_WEIGHTS[45];
    votes[tree_46(x)] += TREE_WEIGHTS[46];
    votes[tree_47(x)] += TREE_WEIGHTS[47];
    votes[tree_48(x)] += TREE_WEIGHTS[48];
    votes[tree_49(x)] += TREE_WEIGHTS[49];
    int best = 0;
    for (int c = 1; c < UAV_CLASS_COUNT; ++c) {
        if (votes[c] > votes[best]) best = c;
    }
    return best;
}
