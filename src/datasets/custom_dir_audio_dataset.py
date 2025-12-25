import torchaudio
from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        """
        Args:
            audio_dir (str): Path to the directory with audio files OR base directory with 'audio' subfolder.
            transcription_dir (str): Path to the directory with transcriptions. 
                                     If None, will look for 'transcriptions' next to 'audio'.
        """
        data = []
        audio_path = Path(audio_dir)

        if (audio_path / "audio").exists():
            transcription_path = audio_path / "transcriptions"
            audio_path = audio_path / "audio"
        else:
            transcription_path = Path(transcription_dir) if transcription_dir else None

        for path in audio_path.iterdir():
            entry = {}
            if path.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path.absolute().resolve())
                if transcription_path and transcription_path.exists():
                    transc_path = transcription_path / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip().lower()


                t_info = torchaudio.info(str(path))
                entry["audio_len"] = t_info.num_frames / t_info.sample_rate

            if "path" in entry:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
