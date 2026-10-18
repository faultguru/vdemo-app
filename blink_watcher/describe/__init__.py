"""The video describer: a clip's sampled key frames -> VideoAnalysis, through
whichever LLM provider is configured (blink_watcher.ai).

  schema     VideoAnalysis, the structured answer
  request    the request built from a frames folder (prompts + images)
  describer  VideoDescriber: sample frames, ask the LLM, save the result
  cli        the `describe` command

Files the describer leaves in a clip's work folder (next to the frames and
manifest.json): the request files of the LLM call (REQUEST_FILES_GLOB, written
when the request is sent) and RESULT_FILE (the answer and the call's details,
written when it is back).
"""

RESULT_FILE = "describe.result.json"
REQUEST_FILES_GLOB = "llm.request*"

__all__ = ["REQUEST_FILES_GLOB", "RESULT_FILE"]
