"""The describer's answer: VideoAnalysis, the structured output schema the
LLM fills for a clip (a summary with people, vehicles, actions and a threat
assessment; one event per moving object per motion segment; the most
informative timestamps and the thumbnail). Stored as the "llm" block of the
processed .metadata.json and the "summary" of the .event.json.

The closed enums are part of the contract with the notification and chat
prompts; the free-form "labels" carry whatever they don't cover.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionType = Literal[
    "package_dropoff",
    "package_pickup",
    "package_carrying",
    "person_walking",
    "person_running",
    "person_standing",
    "person_talking",
    "person_crouching",
    "person_knocking",
    "person_checking_phone",
    "door_opening",
    "gate_opening",
    "mailbox_checking",
    "vehicle_arriving",
    "vehicle_departing",
    "vehicle_passing",
    "person_entering_vehicle",
    "person_exiting_vehicle",
    "dog_walking",
    "animal_present",
    "noise",
]

PersonType = Literal[
    "kid",
    "teen",
    "adult",
    "senior",
    "delivery_person",
    "mail_carrier",
    "police",
    "service_worker",
    "visitor",
    "unknown",
]

ObjectType = Literal[
    "package",
    "phone",
    "bag",
    "box",
    "clipboard",
    "envelope",
    "umbrella",
    "tool",
    "food_container",
    "dog",
    "cat",
    "bicycle",
    "stroller",
]


class Event(BaseModel):
    segment_index: int = Field(description="Which motion segment this event belongs to.")
    object: str = Field(description="Short canonical noun for the moving object: 'person', 'car', 'dog', 'bicycle', 'package', 'animal', 'noise', etc.")
    details: str = Field(description="Distinguishing attributes of this specific object: color, size, clothing, orientation. Empty string if nothing distinctive.")
    description: str = Field(description="One sentence describing what this object is literally doing.")


# Free-form labels next to the closed enums: whatever notable, visible thing
# the enums don't cover (a role, a kind of vehicle, an action, an object).
LABELS_DESCRIPTION = (
    "Free-form labels (short lowercase snake_case) for anything notable and clearly visible that the "
    "fixed enums do not capture — a more specific role, kind of vehicle, action, object or state. "
    "Your own words; empty if the enums already say it all."
)


class Person(BaseModel):
    appearance: str
    appearance_details: str = ""
    type: PersonType = "unknown"
    estimated_age: Optional[int] = None
    actions: list[ActionType] = Field(default_factory=list)
    objects: list[ObjectType] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list, description=LABELS_DESCRIPTION)


class Vehicle(BaseModel):
    appearance: str
    appearance_details: str = ""
    actions: list[ActionType] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list, description=LABELS_DESCRIPTION)


ThreatLevel = Literal["none", "suspicious", "threat"]


class Threat(BaseModel):
    """Threats and suspicious activity, judged only from visible actions and
    objects — never from appearance."""
    level: ThreatLevel = Field(
        default="none",
        description="'threat': danger or a crime clearly in progress (visible weapon held threateningly, "
                    "fighting / assault, forced entry, break-in, vandalism, theft); 'suspicious': behaviour "
                    "that is out of place but not clearly a crime; 'none' otherwise.")
    signals: list[str] = Field(
        default_factory=list,
        description="Your own short lowercase snake_case labels for each visible signal; empty when level is 'none'.")
    evidence: str = Field(
        default="",
        description="One sentence: exactly what is visible and when; empty when level is 'none'.")


class Summary(BaseModel):
    description: str
    actions: list[ActionType] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    vehicles: list[Vehicle] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list, description=LABELS_DESCRIPTION + " For the clip as a whole.")
    threat: Threat = Field(default_factory=Threat, description="Threats and suspicious activity (rule 18).")
    ignore: bool = Field(default=False, description="True only when there is no person or nothing real happens "
                                                    "(rule 20); never for a clip with a person in it.")


class VideoAnalysis(BaseModel):
    summary: Summary = Field(description="Top-level summary of the whole clip.")
    events: list[Event] = Field(
        default_factory=list,
        description="One entry per distinct moving object per segment.",
    )
    important_frames_sec: list[float] = Field(
        default_factory=list,
        description=(
            "Timestamps (seconds from clip start) of the single most informative "
            "moments in the clip — one per distinct action state you want a viewer "
            "to be able to glance at. Typically 2–6 values. Include the thumbnail "
            "moment in this list too. Empty list if there is no real motion."
        ),
    )
    thumbnail_sec: float = Field(
        default=-1.0,
        description=(
            "Single best timestamp (seconds from clip start) to use as the video "
            "thumbnail — the one frame that captures what the clip is about. "
            "Should be one of the values in important_frames_sec. Use -1.0 if "
            "there is no real motion."
        ),
    )
