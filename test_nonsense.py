from vieneu import Vieneu
import os
import numpy as np
from loguru import logger
import soundfile as sf

def test_split_segments():
    try:
        tts = Vieneu()
        logger.info("VieNeu initialized")
        
        # Original long text
        full_text = "Bây giờ tôi đang ở Venus, cùng với đối tác của mình, chúng tôi đang ở Venus nhìn quanh những con phố tuyệt vời, tôi thích nó, nó thật hoàn hảo, nhìn kìa, đẹp quá, tôi nóng lòng muốn cho các bạn xem những chiếc bánh pizza."
        
        # Split into short segments (simulating translation.json)
        segments = [
            "Bây giờ tôi đang ở Venus, cùng với đối tác của mình,",
            "chúng tôi đang ở Venus nhìn quanh những con phố tuyệt vời,",
            "tôi thích nó, nó thật hoàn hảo,",
            "nhìn kìa, đẹp quá,",
            "tôi nóng lòng muốn cho các bạn xem những chiếc bánh pizza."
        ]
        
        # Original reference audio (if available, otherwise use a placeholder or None)
        # We'll try to find a real wav files to make it realistic
        ref_audio = "outputs/studio_grade/video3/audio_vocals.wav"
        if not os.path.exists(ref_audio):
             logger.warning(f"Ref audio {ref_audio} not found, using None (Default voice mode? or Error?)")
             ref_audio = None
        
        logger.info(f"Using Ref Audio: {ref_audio}")

        for i, seg_text in enumerate(segments):
            logger.info(f"--- Segment {i+1} ---")
            logger.info(f"Text: {seg_text}")
            logger.info(f"Length: {len(seg_text)} chars")
            
            output_path = f"test_seg_{i}.wav"
            
            # Generate
            # We use the full ref_audio for each small segment to reproduce the "nonsense" issue
            audio = tts.infer(
                text=seg_text,
                ref_audio=ref_audio,
                ref_text=full_text, # Passing FULL text as prompt (what we do now) vs segment text
                temperature=0.7 
            )
            
            tts.save(audio, output_path)
            
            # Check duration
            if os.path.exists(output_path):
                 dur = len(audio) / 24000
                 logger.info(f"Generated: {dur:.2f}s | Saved to {output_path}")
            else:
                 logger.error("Failed to generate file")

    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_split_segments()
