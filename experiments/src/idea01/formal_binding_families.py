"""Frozen author designs for the formal geometry-only Table 4 fixture bank."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import re

from . import in_domain_pilot_geometry as pilot


SEED = "idea01-table4-author-20261124-v1"
SOURCE = "Original procedural CAD constructed for this study on 2026-11-24"
EXCLUDED_DEVELOPMENT_FAMILIES = tuple(pilot.FAMILIES)
ATTRIBUTES = tuple(pilot.ATTRIBUTES)
KINDS = ("binding", "easy")
PRIMARY_COUNT = 20
BACKUP_COUNT = 2
COORDINATE_COUNT = 10
HISTORIES = pilot.HISTORIES
CAPTION_VARIANTS = (
    "near_first", "far_first", "near_first_rephrased", "far_first_rephrased",
)
NEAR_FAR = ("near", "far")


@dataclass(frozen=True)
class Family:
    part: str
    anchor: str
    tree: tuple[str, ...]
    topology: str
    measure: str
    hole_group: str = "vertical through holes"


FAMILIES = {
    "end_eye_link": Family(
        "eye-ended link", "center of the large end eye",
        ("extrude rectangular link shank", "join circular end lobe", "cut closed through eye"),
        "Circular stock lobe and a third through bore reserved for the link eye.",
        "Projected center of the unique largest circular through opening.",
        "vertical through holes other than the large end eye",
    ),
    "blind_keyseat_block": Family(
        "keyseat block", "center of the rectangular recess floor",
        ("extrude deep rectangular blank", "cut rectangular blind keyseat"),
        "Enclosed four-wall rectangular blind cavity with a planar floor.",
        "Centroid of the unique upward intermediate horizontal quadrilateral.",
    ),
    "dual_skid_plate": Family(
        "skid plate", "center of the longer underside skid",
        ("extrude raised plate", "join long underside skid", "join short underside skid"),
        "Two disjoint coplanar bottom supports joined through a shared raised deck.",
        "Centroid of the larger of the two bottommost downward planar faces.",
    ),
    "returned_corner_bracket": Family(
        "returned corner bracket", "inside corner at the top of the raised flange",
        ("extrude rectangular foot", "join a returned flange from its horizontal L-shaped footprint"),
        "Two perpendicular raised walls share an inside corner above the open deck.",
        "Unique reflex vertex on the upper perimeter of the returned flange.",
    ),
    "side_window_portal": Family(
        "portal bracket", "center of the rectangular side window",
        ("extrude foot", "join upright rectangular frame blank", "cut transverse rectangular window"),
        "A four-sided transverse through window joins two vertical openings above the deck.",
        "Midpoint between the centroids of the two opposed four-edge side openings.",
    ),
    "transverse_blind_lug": Family(
        "side-bored lug bracket", "center of the blind side bore opening",
        ("extrude foot", "join one raised lug", "cut transverse blind circular bore"),
        "A horizontal cylindrical cavity has one side mouth and a closed circular cap.",
        "Centroid of the unique circular inner wire on a vertical outer face.",
    ),
    "dovetail_saddle": Family(
        "dovetail saddle", "center of the floor between the sloping jaws",
        ("extrude constant side profile with two re-entrant sloped jaws",),
        "Open-ended undercut channel between two tapered jaws above a full-width floor.",
        "Centroid of the unique upward intermediate horizontal face above the deck.",
    ),
    "cantilever_lip_plate": Family(
        "cantilever lip plate", "center of the underside of the overhanging lip",
        ("extrude hooked side profile including deck, web and cantilevered lip",),
        "Re-entrant lateral relief under a single unsupported overhanging lip.",
        "Centroid of the unique downward horizontal face above the global bottom.",
    ),
    "pointed_mounting_web": Family(
        "pointed triangular mounting web", "pointed tip",
        ("extrude three-edge triangular web",),
        "Convex triangular perimeter with a single acute longitudinal apex.",
        "Unique sharpest vertex of the upper triangular outer perimeter.",
    ),
    "two_level_terrace": Family(
        "two-level terrace", "midpoint of the upper edge of the step",
        ("extrude full-width two-deck stair section",),
        "A full-width riser separates two decks pierced at different heights.",
        "Midpoint of the unique upper-deck boundary edge interior in plan.",
    ),
    "annular_socket_plate": Family(
        "socket plate", "center of the circular socket on the raised collar",
        ("extrude plate", "join raised round collar stock", "cut axial blind socket"),
        "A raised coaxial collar has an annular top and a recessed blind socket floor.",
        "Center of the unique annular top face, checked from the two concentric wires.",
    ),
    "inclined_gusset_plate": Family(
        "inclined gusset plate", "midpoint of the highest edge of the gusset",
        ("extrude foot", "join narrow triangular extrusion from transverse side profile"),
        "A narrow inclined triangular rib ends at a ridge, with open deck on both sides.",
        "Midpoint of the unique highest straight edge.",
    ),
}


def seed_number(*values) -> int:
    key = "|".join(map(str, (SEED, *values)))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Slot:
    family: str
    attribute: str
    kind: str
    index: int
    quarter_turns: int
    a_marked: bool
    first0_near: bool
    view_order: int
    view_variant: str
    easy_equal_high: bool
    coordinate: bool

    @property
    def id(self) -> str:
        return f"{self.family}/{self.attribute}/{self.kind}/{self.index:02d}"


def slots():
    """Balance 5 per quarter-turn and 10 per binary factor, before inspecting any solid."""
    for family in FAMILIES:
        for attribute in ATTRIBUTES:
            for kind in KINDS:
                cell = (family, attribute, kind)
                assignments = {}
                for block in range(5):
                    # All four orientation/cut-label cells occur in every block.
                    order = sorted(range(4), key=lambda value: seed_number(*cell, "axes", block, value))
                    phases = seed_number(*cell, "phases", block)
                    for offset, quadrant in enumerate(order):
                        low, high = quadrant & 1, quadrant >> 1
                        a_marked = bool(high ^ (phases & 1))
                        first0_near = bool(low ^ ((phases >> 1) & 1))
                        view_order = high ^ ((phases >> 2) & 1)
                        caption_near = low ^ high ^ ((phases >> 3) & 1)
                        equal_high = bool(low ^ ((phases >> 4) & 1))
                        variant = "near_first" if caption_near else "far_first"
                        index = 4 * block + offset
                        assignments[index] = (
                            quadrant, a_marked, first0_near, view_order, variant, equal_high,
                        )
                coordinate = set()
                if kind == "binding":
                    for a_marked in (False, True):
                        for first_near in (False, True):
                            ordered = sorted(
                                (i for i, params in assignments.items() if params[1:3] == (a_marked, first_near)),
                                key=lambda i: seed_number(*cell, "coordinate", i),
                            )
                            coordinate.update(ordered[:3 if a_marked == first_near else 2])
                for index, params in assignments.items():
                    yield Slot(*cell, index, *params, index in coordinate)


@dataclass(frozen=True)
class Dimensions:
    width: int
    height: int
    deck: int
    relief: int
    near: tuple[int, int]
    far: tuple[int, int]
    large: int
    small: int
    half_straight: int


@dataclass(frozen=True)
class Design:
    slot: Slot
    rank: int
    dimensions: Dimensions

    @property
    def pair_id(self) -> str:
        return self.slot.id.replace("/", "-") + f"-r{self.rank}"

    @property
    def family(self) -> str:
        return self.slot.family

    def expected(self, entity: str) -> tuple[str, str] | str:
        marked = (entity == "A") == self.slot.a_marked
        if self.slot.kind == "easy":
            return "different" if marked else "same"
        forms = pilot.ATTRIBUTES[self.slot.attribute]
        return forms if marked else forms[::-1]

    def holes(self, entity: str) -> tuple[pilot.Hole, pilot.Hole]:
        d = self.dimensions
        size = self.slot.attribute == "size_location"
        forms = ((d.large, 0), (d.small, 0)) if size else ((d.small, 0), (d.small, d.half_straight))
        expected = self.expected(entity)
        if self.slot.kind == "binding":
            indices = (0, 1) if expected == pilot.ATTRIBUTES[self.slot.attribute] else (1, 0)
        elif expected == "same":
            equal_index = int(not self.slot.easy_equal_high) if size else int(self.slot.easy_equal_high)
            indices = (equal_index, equal_index)
        else:
            indices = (0, 1) if self.slot.easy_equal_high else (1, 0)
        return tuple(pilot.Hole(center, *forms[index])
                     for center, index in zip((d.near, d.far), indices, strict=True))

    def rotate_xy(self, xy):
        return rotate_xy(xy, self.slot.quarter_turns)


def rotate_xy(xy, quarter_turns):
    x, y = xy
    return ((x, y), (-y, x), (-x, -y), (y, -x))[quarter_turns]


def design_at(slot: Slot, rank: int) -> Design:
    if not 0 <= rank <= BACKUP_COUNT:
        raise ValueError("invalid frozen backup rank")
    dimensions = [seed_number(slot.id, rank, "dimension", i) for i in range(8)]
    width = 192 + slot.index + 4 * rank
    height = 108 + 2 * (dimensions[0] % 6)
    y = height // 2 + dimensions[1] % 7 - 3
    small = 7 + dimensions[5] % 3
    return Design(slot, rank, Dimensions(
        width, height, 14 + dimensions[2] % 5, dimensions[3] % 5,
        (89 + dimensions[4] % 4, y), (width - 30 - dimensions[6] % 3, y),
        small + 5, small, 11 + dimensions[7] % 2,
    ))


@dataclass(frozen=True)
class Profile:
    polygon: tuple[tuple[int, int], ...] = ()
    hole: pilot.Hole | None = None
    quarter_turns: int = 0

    def world(self, point):
        return rotate_xy(point, self.quarter_turns)


@dataclass(frozen=True)
class Feature:
    profile: Profile
    depth: int
    operation: str
    origin: tuple[int, int, int] = (0, 0, 0)
    x_axis: tuple[int, int, int] = (1, 0, 0)
    y_axis: tuple[int, int, int] = (0, 1, 0)

    @property
    def z_axis(self):
        ax, ay, az = self.x_axis
        bx, by, bz = self.y_axis
        return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    def global_point(self, point, depth=0):
        x, y = point
        return tuple(o + x * u + y * v + depth * n for o, u, v, n in zip(
            self.origin, self.x_axis, self.y_axis, self.z_axis, strict=True,
        ))


CREATE = "NewBodyFeatureOperation"
JOIN = "JoinFeatureOperation"
CUT = "CutFeatureOperation"


def polygon(points, depth, operation=CREATE, *, z=0):
    return Feature(Profile(polygon=tuple(points)), depth, operation, origin=(0, 0, z))


def rectangle(x0, x1, y0, y1, z, depth, operation=CREATE):
    return polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), depth, operation, z=z)


def cylinder(x, y, radius, z, depth, operation):
    return Feature(Profile(hole=pilot.Hole((x, y), radius)), depth, operation, (0, 0, z))


def side_profile(points, y, depth, operation=CREATE):
    return Feature(Profile(polygon=tuple(points)), depth, operation, (0, y, 0), (1, 0, 0), (0, 0, 1))


def stock_features(design: Design) -> tuple[Feature, ...]:
    """A construction recipe in local author units, prior to the balanced world pose."""
    family = design.family
    w, h, d, j = (getattr(design.dimensions, name) for name in ("width", "height", "deck", "relief"))
    cy = h // 2
    base = rectangle(0, w, 0, h, 0, d)
    if family == "end_eye_link":
        return (rectangle(49, w, 0, h, 0, d), cylinder(37, cy, 34 + j, 0, d, JOIN),
                cylinder(37, cy, 19 + j % 2, -4, d + 8, CUT))
    if family == "blind_keyseat_block":
        return (rectangle(0, w, 0, h, 0, d + 12),
                rectangle(10, 54 + j, cy - 23, cy + 23, d, 16, CUT))
    if family == "dual_skid_plate":
        return (rectangle(0, w, 0, h, 20, d),
                rectangle(12, 35, 12, h - 12, 0, 23, JOIN),
                rectangle(43, 64, 12, cy - 7, 0, 23, JOIN))
    if family == "returned_corner_bracket":
        footprint = ((0, 0), (73, 0), (73, 20), (36, 20), (36, h), (0, h))
        return (base, polygon(footprint, 46 + j, JOIN, z=d - 4))
    if family == "side_window_portal":
        top = d + 52 + j
        opening = Profile(polygon=((24, d + 12), (h - 24, d + 12), (h - 24, top - 12), (24, top - 12)))
        return (base, rectangle(15, 43, 10, h - 10, d - 4, top - d + 4, JOIN),
                Feature(opening, 37, CUT, (10, 0, 0), (0, 1, 0), (0, 0, 1)))
    if family == "transverse_blind_lug":
        bore = Profile(hole=pilot.Hole((cy + 2, d + 22), 12 + j % 2))
        return (base, rectangle(14, 48, cy - 24, cy + 26, d - 4, 44 + j, JOIN),
                Feature(bore, 31, CUT, (5, 0, 0), (0, 1, 0), (0, 0, 1)))
    if family == "dovetail_saddle":
        top = d + 30 + j
        outline = ((0, 0), (w, 0), (w, d), (59, d), (59, top), (39, top),
                   (45, d + 9), (22, d + 9), (28, top), (9, top), (9, d), (0, d))
        return (side_profile(outline, h, h),)
    if family == "cantilever_lip_plate":
        bottom, top = d + 22, d + 35 + j
        outline = ((0, 0), (w, 0), (w, d), (54, d), (54, top), (4, top),
                   (4, bottom), (37, bottom), (37, d), (0, d))
        return (side_profile(outline, h, h),)
    if family == "pointed_mounting_web":
        return (polygon(((0, cy), (w, 0), (w, h)), d),)
    if family == "two_level_terrace":
        top = d + 22 + j
        return (side_profile(((0, 0), (w, 0), (w, d), (118, d), (118, top), (0, top)), h, h),)
    if family == "annular_socket_plate":
        return (base, cylinder(35, cy, 27 + j, d - 3, 39 + j, JOIN),
                cylinder(35, cy, 14 + j % 2, d + 5, 35 + j, CUT))
    if family == "inclined_gusset_plate":
        rib = ((10, d - 4), (62, d - 4), (17, d + 40 + j))
        return (base, side_profile(rib, cy + 14, 28 + j, JOIN))
    raise ValueError(f"Unknown formal family {family}")


def rotate_feature(design: Design, feature: Feature) -> Feature:
    def rotate(vector):
        return (*design.rotate_xy(vector[:2]), vector[2])

    if feature.z_axis == (0, 0, 1):
        # For the normal panel, keep horizontal sketch frames aligned with world XY.
        # Bake the world pose into the profile, as in the development geometry.
        profile = feature.profile
        return Feature(Profile(profile.polygon, profile.hole, design.slot.quarter_turns),
                       feature.depth, feature.operation, rotate(feature.origin))
    return Feature(feature.profile, feature.depth, feature.operation, rotate(feature.origin),
                   rotate(feature.x_axis), rotate(feature.y_axis))


def history_features(design: Design, member: str, coordinate=False):
    if member not in HISTORIES:
        raise ValueError(f"unknown history {member}")
    stock = [rotate_feature(design, feature) for feature in stock_features(design)]
    top = solid_bounds(stock)[5]
    cuts = []
    for role, hole in zip(NEAR_FAR, design.holes(member[0]), strict=True):
        x, y = design.rotate_xy(hole.center)
        # Rotate around each cutter's center so the polygon stays in world XY.
        quarter_turns = design.slot.quarter_turns
        u, v = (1, 0, 0), (0, 1, 0)
        cx, cy = x, y
        ox, oy = 0, 0
        if coordinate:
            d = design.dimensions
            dx, dy = (d.near[i] + d.far[i] for i in range(2))
            ox, oy = design.rotate_xy((dx, dy))
            cx, cy = ox - cx, oy - cy
            u, v = tuple(-value for value in u), tuple(-value for value in v)
        local_center = rotate_xy((cx, cy), (-quarter_turns) % 4)
        cut = Feature(Profile(hole=pilot.Hole(local_center, hole.radius, hole.half_straight),
                              quarter_turns=quarter_turns),
                      top + 8, CUT, (ox, oy, -4), u, v)
        if cut.global_point((cx, cy))[:2] != (x, y):
            raise ValueError("compensated local frame moved the hole center")
        cuts.append((role, cut))
    if design.slot.first0_near != (member[1] == "0"):
        cuts.reverse()
    return stock + [feature for _, feature in cuts], tuple(role for role, _ in cuts)


def solid_bounds(features):
    """Exact envelope of the additive axis-aligned sketches and their positive extrusions."""
    points = []
    for feature in features:
        if feature.operation == CUT:
            continue
        profile = feature.profile
        if profile.hole:
            x, y = profile.hole.center
            r = profile.hole.radius
            perimeter = ((x - r, y - r), (x + r, y + r))
            # Include all four corners as the local-to-world frame can reverse one axis.
            perimeter += ((x - r, y + r), (x + r, y - r))
        else:
            perimeter = profile.polygon
        points.extend(feature.global_point(profile.world(point), depth)
                      for point in perimeter for depth in (0, feature.depth))
    axes = tuple(zip(*points, strict=True))
    return tuple(min(axis) for axis in axes) + tuple(max(axis) for axis in axes)


def _xyz(values):
    return dict(zip("xyz", values, strict=True))


def profile_curves(profile: Profile) -> list[dict]:
    if profile.hole:
        return pilot._hole_curves(profile, profile.hole)
    points = tuple(profile.world(point) for point in profile.polygon)
    return [pilot._line(start, end) for start, end in zip(points, points[1:] + points[:1], strict=True)]


def raw_program(features, bounds=None) -> dict:
    if bounds is None:
        bounds = solid_bounds(features)
    sequence = []
    entities = {}
    for index, feature in enumerate(features, start=1):
        sketch, extrude = f"sketch_{index}", f"extrude_{index}"
        entities[sketch] = {
            "type": "Sketch",
            "transform": {"origin": _xyz(feature.origin), "x_axis": _xyz(feature.x_axis),
                          "y_axis": _xyz(feature.y_axis), "z_axis": _xyz(feature.z_axis)},
            "profiles": {"profile_1": {"loops": [
                {"is_outer": True, "profile_curves": profile_curves(feature.profile)},
            ]}},
        }
        entities[extrude] = {
            "type": "ExtrudeFeature", "profiles": [{"sketch": sketch, "profile": "profile_1"}],
            "start_extent": {"type": "ProfilePlaneStartDefinition"},
            "extent_type": "OneSideFeatureExtentType", "extent_one": {"distance": {"value": feature.depth}},
            "operation": feature.operation,
        }
        sequence.extend(({"type": "Sketch", "entity": sketch}, {"type": "ExtrudeFeature", "entity": extrude}))
    return {"sequence": sequence, "entities": entities, "properties": {
        "bounding_box": {"min_point": _xyz(bounds[:3]), "max_point": _xyz(bounds[3:])},
    }}


DRAFT_PARAPHRASES = {
    "near_first_rephrased": (
        "In the {part}, the {near} opening has its center closest to the {anchor} in plan view. "
        "The center of the {far} opening is at greater distance. Both openings are {holes}."
    ),
    "far_first_rephrased": (
        "In the {part}, the center of the {far} opening is at greater plan-view distance from the {anchor}. "
        "The center of the {near} opening is closer. Both openings are {holes}."
    ),
}


def captions(design: Design, paraphrases: dict) -> dict[str, dict[str, str]]:
    family = FAMILIES[design.family]
    common = {"part": family.part, "anchor": family.anchor, "holes": family.hole_group}
    result = {}
    for entity in ("A", "B"):
        expected = design.expected(entity)
        if design.slot.kind == "easy":
            noun = "diameter" if design.slot.attribute == "size_location" else "shape"
            result[entity] = {
                "near_first": (
                    f"In the {family.part}, the {family.hole_group} nearer to and farther from "
                    f"the {family.anchor} in plan view have the {expected} {noun}."
                ),
                "far_first": (
                    f"In the {family.part}, the {family.hole_group} farther from and nearer to "
                    f"the {family.anchor} in plan view have the {expected} {noun}."
                ),
            }
            continue
        near, far = expected
        front = f"Consider the {family.hole_group} of the {family.part} in plan view."
        result[entity] = {
            "near_first": (
                f"{front} The hole with its center nearer to the {family.anchor} is {near}. "
                f"The hole with its center farther away is {far}."
            ),
            "far_first": (
                f"{front} The hole with its center farther from the {family.anchor} is {far}. "
                f"The hole with its center nearer is {near}."
            ),
            **{variant: paraphrases[variant].format(near=near, far=far, **common)
               for variant in CAPTION_VARIANTS[2:]},
        }
    tokens = lambda value: Counter(re.findall(r"[a-z0-9]+", value.lower()))
    for variant in result["A"]:
        a, b = result["A"][variant], result["B"][variant]
        if a == b or (design.slot.kind == "binding" and tokens(a) != tokens(b)):
            raise ValueError(f"opposed captions failed the vocabulary check in {variant}")
    if design.slot.kind == "binding" and any(len(set(item.values())) != 4 for item in result.values()):
        raise ValueError("binding caption variants must all be distinct")
    return result


def check_design_balance(all_slots):
    for family in FAMILIES:
        for attribute in ATTRIBUTES:
            for kind in KINDS:
                group = [slot for slot in all_slots if (slot.family, slot.attribute, slot.kind) == (family, attribute, kind)]
                axes = Counter(slot.quarter_turns for slot in group)
                if len(group) != PRIMARY_COUNT or axes != dict.fromkeys(range(4), 5):
                    raise ValueError("invalid orientation balance")
                for field in ("a_marked", "first0_near", "view_order", "view_variant", "easy_equal_high"):
                    if sorted(Counter(getattr(slot, field) for slot in group).values()) != [10, 10]:
                        raise ValueError(f"invalid binary balance for {field}")
                if sum(slot.first0_near != bool(slot.view_order) for slot in group) != 10:
                    raise ValueError("invalid selected first-cut balance")
                if Counter((slot.view_order, slot.view_variant) for slot in group) != dict.fromkeys(
                    ((0, "near_first"), (0, "far_first"), (1, "near_first"), (1, "far_first")), 5
                ):
                    raise ValueError("invalid selected program/narration balance")
                expected = COORDINATE_COUNT if kind == "binding" else 0
                if sum(slot.coordinate for slot in group) != expected:
                    raise ValueError("invalid coordinate subset size")
