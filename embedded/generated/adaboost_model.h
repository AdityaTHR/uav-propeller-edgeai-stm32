#ifndef ADABOOST_MODEL_H
#define ADABOOST_MODEL_H

#define UAV_FEATURE_COUNT 16
#define UAV_CLASS_COUNT 5
#define UAV_TREE_COUNT 50

int adaboost_predict(const float x[UAV_FEATURE_COUNT]);

#endif
