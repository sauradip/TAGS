from evaluation.eval_detection import ANETdetection

print("Evaluation Started")
anet_detection = ANETdetection(
    ground_truth_filename="./evaluation/activity_net_1_3_new.json",
    # prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
    prediction_filename="output_TAGS.json",
    subset='validation', verbose=False, check_status=False)
anet_detection.evaluate()
print("Evaluation Finished")
mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
print(results)
