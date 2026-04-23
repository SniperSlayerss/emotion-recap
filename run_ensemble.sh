# python arousal_detection/train_ensemble.py training_sessions/ --label-prefix baseline \
#     --gsr-contamination 0.05 --hrv-contamination 0.05 \
#     --out models/ensemble

python arousal_detection/tune_threshold_ensemble.py cut_sessions/ \
    --baseline-prefix baseline \
    --contaminations 0.005,0.01,0.02,0.05,0.075,0.1,0.15


python arousal_detection/analyse_results.py cut_sessions/ \
    --model models/ensemble \
    --out results/

# python detect_from_session.py training_sessions/<session> \
#     --model models/ensemble --combine-mode any

# python collect_training_data.py 0 --label horror_film \
#     --model models/ensemble --combine-mode any
