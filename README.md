# Assignment 3 - SeqTrack Setup, Training, and Checkpoint Management

## Submission Package Contents

### 📋 Main Documents
- **assignment_3.docx**: Complete assignment report with all requirements addressed
- **requirements.txt**: Python package dependencies
- **installed_packages.txt**: Complete list of installed packages

### 📊 Results & Analysis
- **loss_phase1_phase2.png**: Training loss comparison graph
- **iou_phase1_phase2.png**: Validation IoU comparison graph
- **metrics.json**: Numerical metrics in JSON format
- **phase1_vs_phase2_detailed_analysis.md**: Detailed comparison analysis

### 📝 Training Logs
- **training_log.txt**: Phase 1 complete training logs (epochs 1-10)
- **training_log_phase2.txt**: Phase 2 training logs (epochs 4-10)

### 💻 Source Code
- **utils.py**: Core utility functions (seed, logging, checkpoint save/load)
- **train.py**: Phase 1 training script
- **train_phase2.py**: Phase 2 training script (resume from epoch 3)

## Key Results

### Classes Used
- electricfan (20 sequences)
- mouse (20 sequences)

### Training Configuration
- Samples per epoch: 7,000 (training), 1,000 (validation)
- Total epochs: 10
- Fixed seed per epoch: Team number = 2

### Final Results (Epoch 10)
- Training Loss: 6.4229
- Validation IoU: 0.6400

### Phase 1 vs Phase 2 Verification
✅ **PERFECT MATCH**: All loss and IoU values for epochs 4-10 are identical between Phase 1 and Phase 2, confirming correct checkpoint implementation.

## Repository
- GitHub: https://github.com/Abdalrhman-m/S2S_Assignment-3
- Branch: master

## Checkpoints
- **Phase 1**: Saved locally in `checkpoints/` and uploaded to Hugging Face
  - Repository: https://huggingface.co/Medo-2004/assignment_3
- **Phase 2**: Saved locally only in `checkpoints_phase2/`

## Assignment Requirements Fulfilled

✅ 1. Environment setup complete (VideoX/SeqTrack repository cloned and configured)
✅ 2. LaSOT dataset prepared with 2 classes selected
✅ 3. Checkpoints include optimizer, scheduler, and RNG states
✅ 4. Trained for 10 epochs (Phase 1)
✅ 5. Fixed seed (team number) set at beginning of each epoch
✅ 6. Checkpoint saved at end of every epoch (10 total)
✅ 7. Automatic upload to Hugging Face configured (Phase 1)
✅ 8. Training resumption from checkpoint implemented
✅ 9. Re-run training from epoch 3 to 10 (Phase 2)
✅ 10. Training logs with time statistics every 50 samples
✅ 11. All log information printed to screen and file
✅ 12. Class names and dataset sizes documented
✅ 13. Complete package list provided
✅ 14. Code modifications listed with file/line numbers
✅ 15. Loss and IoU graphs generated
✅ 16. Phase 1 and Phase 2 results verified as identical
✅ 17. GitHub repository link provided

## Contact
Team: S2S Assignment-3
Repository Owner: Abdalrhman-m
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/275f5a17-ca0e-43ae-8985-0957e495e46d" />

