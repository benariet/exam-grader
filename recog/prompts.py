"""The prompts of the vision-language readers (recog/hebrew_vlm.py, recog/mumer.py) - kept apart so code that
only needs a prompt does not load torch."""

PROMPT = ("Transcribe the handwritten Hebrew text in this image exactly as written. "
          "Output only the Hebrew text, in logical (right-to-left reading) order, with no explanations.")
PROMPT_ENGLISH = ("Transcribe the handwritten English text in this image exactly as written, keeping its spelling "
                  "mistakes. Output only the text, with no explanations.")
PROMPT_MATH = ("I have an image of a handwritten mathematical expression. Please write out the expression of the "
               "formula in the image using LaTeX format.")          # Uni-MuMER's own prompt (arXiv 2505.23566, A.1)
