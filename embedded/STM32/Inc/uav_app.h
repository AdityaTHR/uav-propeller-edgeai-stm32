#ifndef UAV_APP_H
#define UAV_APP_H

int uav_app_init(void);
int uav_predict_window(const float *x, const float *y, float *features_out);
const char *uav_class_name(int class_id);

#if UAV_SELF_TEST
int uav_run_self_test(void);
#endif

#endif
