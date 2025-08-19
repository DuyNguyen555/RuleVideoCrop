import cv2

def draw_thresholds(fr, thresholds, bar_p):
    y_up_above, y_up_below = thresholds
    cv2.line(fr, (0, int(y_up_above)), (fr.shape[1], int(y_up_above)), (0,255,0), 2)
    cv2.putText(fr, f"up_above:{int(y_up_above)}", (10, int(y_up_above)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.line(fr, (0, int(y_up_below)), (fr.shape[1], int(y_up_below)), (0,128,255), 2)
    cv2.putText(fr, f"up_below:{int(y_up_below)}", (10, int(y_up_below)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)

    cv2.line(fr, (0, int(bar_p)), (fr.shape[1], int(bar_p)), (0,0,255), 2)
    cv2.putText(fr, f"bar_p:{int(bar_p)}", (10, int(bar_p)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return fr
