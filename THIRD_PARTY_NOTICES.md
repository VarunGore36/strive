# Third-party components and model provenance

The repository includes original application code and uses externally installed packages. It does not bundle third-party speech checkpoints, datasets, recorded voices or the uploaded source documents. No project-wide license is selected on the team's behalf; choose one before presenting the source as an open-source release.

NII model preparation/export follows the published [AntiDeepfake model documentation](https://github.com/nii-yamagishilab/AntiDeepfake). Its checkpoint and code licenses differ: checkpoint CC BY-NC-SA 4.0, code BSD-3-Clause. Preserve upstream notices for any third-party code or weights you later distribute. Download scripts do not alter upstream licenses.

Other model references:

- [Meta XLSR phoneme model](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
- [SpeechBrain VoxLingua language model](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- [Vox-Profile](https://github.com/tiantiaf0627/vox-profile-release)

Review each specific checkpoint's license and intended use before government or commercial deployment. The demo requires no paid API; hardware, network, storage and operational costs still exist.

The frontend bundles the Unbounded typeface from Google Fonts under the SIL Open Font License 1.1. The complete license text is stored at `web/fonts/OFL-Unbounded.txt`.
