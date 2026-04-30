# python arousal_detection/tune_threshold_ensemble.py clean_sessions_only_horror/ \
#     --baseline-prefix baseline \
#     --contaminations 0.005,0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4


# python detect_from_session.py training_sessions/<session> \
#     --model models/ensemble --combine-mode any

# python collect_training_data.py 0 --label horror_film \
#     --model models/ensemble --combine-mode any

python arousal_detection/train_ensemble.py clean_sessions/ --label-prefix baseline \
    --gsr-contamination 0.02 --hrv-contamination 0.075 \
    --out models/ensemble

python arousal_detection/analyse_results.py clean_sessions/ --model models/ensemble --out results/mode_any  --combine-mode any --operating-threshold 0.22
python arousal_detection/analyse_results.py clean_sessions/ --model models/ensemble --out results/mode_all  --combine-mode all --operating-threshold 0.22
python arousal_detection/analyse_results.py clean_sessions/ --model models/ensemble --out results/mode_mean --combine-mode mean --operating-threshold 0.22

# python arousal_detection/train_ensemble.py clean_sessions_only_horror/ --label-prefix baseline \
#     --gsr-contamination 0.02 --hrv-contamination 0.075 \
#     --out models/ensemble
#
# python arousal_detection/analyse_results.py clean_sessions_only_horror/ --model models/ensemble --out results/mode_any  --combine-mode any --operating-threshold 0.22
# python arousal_detection/analyse_results.py clean_sessions_only_horror/ --model models/ensemble --out results/mode_all  --combine-mode all --operating-threshold 0.22
# python arousal_detection/analyse_results.py clean_sessions_only_horror/ --model models/ensemble --out results/mode_mean --combine-mode mean --operating-threshold 0.22
