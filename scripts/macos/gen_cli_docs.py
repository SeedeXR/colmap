#!/usr/bin/env python3
"""Generate exhaustive CLI docs for the COLMAP mac build by driving the binary.

Produces, under output/:
  man/colmap.1            (top-level man page, copied from doc/colmap.1)
  man/colmap-<cmd>.1      (one man page per subcommand, from `colmap <cmd> -h`)
  llm.txt                 (dense, LLM-oriented full reference)
  README.md               (human-readable, exhaustive reference)

Option lists are read live from the binary, so they cannot drift from reality.
"""

import datetime
import os
import re
import subprocess
import sys

# Paths derive from this script's location (repo = ../..) so docs can be
# regenerated from any checkout. Override the binary with argv[1] or $COLMAP_REPO.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("COLMAP_REPO") or os.path.abspath(
    os.path.join(HERE, "..", "..")
)
BIN = (
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "colmap")
)
OUT = os.path.join(REPO, "output")
MAN = os.path.join(OUT, "man")
os.makedirs(MAN, exist_ok=True)

DATE = datetime.date.today().strftime("%B %Y")
ISO = datetime.date.today().isoformat()


def run(args):
    p = subprocess.run([BIN] + args, capture_output=True, text=True)
    return p.stdout + p.stderr


VERSION = run(["version"]).strip().splitlines()[0]

# ---- Parse the grouped command list from `colmap help` -------------------
help_txt = run(["help"])
GROUPS = []  # [(group_name, [cmd, ...])]
cur_group = None
in_cmds = False
for ln in help_txt.splitlines():
    if ln.strip() == "Available commands:":
        in_cmds = True
        continue
    if not in_cmds:
        continue
    m_group = re.match(r"^  ([A-Z][^:]*):\s*$", ln)
    m_cmd = re.match(r"^    ([a-z][a-z0-9_]*)\s*$", ln)
    if m_group:
        cur_group = m_group.group(1).strip()
        GROUPS.append((cur_group, []))
    elif m_cmd and GROUPS:
        GROUPS[-1][1].append(m_cmd.group(1))

ALL_CMDS = [c for _, cmds in GROUPS for c in cmds]

# ---- Authored one-line descriptions (factual; option detail comes live) --
DESC = {
    "help": "List all commands (grouped) with example usage.",
    "version": "Print the COLMAP version and build configuration.",
    "gui": "Launch the Qt graphical interface. NOT available in this build "
    "(compiled with GUI_ENABLED=OFF).",
    "automatic_reconstructor": "End-to-end pipeline from an image folder: "
    "extraction, matching, sparse mapping, and (only with CUDA) dense MVS.",
    "project_generator": "Write template .ini project files for the pipeline "
    "commands.",
    "mapper_advisor": "Inspect a database and advise on mapper settings/strategy.",
    "system_info": "Print machine resources (RAM, performance/efficiency CPU "
    "cores) as text or JSON; used for resource-aware tuning on this build.",
    "database_creator": "Create an empty COLMAP SQLite database.",
    "database_cleaner": "Delete entries (images, features, and/or matches) from "
    "a database.",
    "database_merger": "Merge two databases into one.",
    "feature_extractor": "Detect and describe features and store them in the "
    "database. This build uses CPU SIFT (VLFeat) or ALIKED (ONNX); GPU SIFT "
    "is unavailable here (it needs CUDA or OpenGL), so use_gpu has no effect.",
    "feature_importer": "Import precomputed keypoints/descriptors from files "
    "into the database.",
    "exhaustive_matcher": "Match all image pairs (best for small/medium "
    "unordered collections).",
    "sequential_matcher": "Match consecutive frames for ordered image/video "
    "sequences, with optional vocab-tree loop closure.",
    "spatial_matcher": "Match images by spatial proximity using location priors.",
    "transitive_matcher": "Add image-pair matches transitively from existing "
    "matches.",
    "vocab_tree_matcher": "Match images using a vocabulary tree (best for large "
    "unordered collections).",
    "vocab_tree_builder": "Build a vocabulary tree from a database of features.",
    "vocab_tree_retriever": "Retrieve visually similar images via a vocabulary "
    "tree.",
    "matches_importer": "Import precomputed image-pair matches from a file.",
    "geometric_verifier": "Run geometric verification on existing putative "
    "matches.",
    "guided_geometric_verifier": "Geometric verification guided by a prior "
    "two-view geometry/model.",
    "mapper": "Incremental Structure-from-Motion: the core sparse reconstruction "
    "engine.",
    "global_mapper": "Global Structure-from-Motion (rotation averaging + global "
    "positioning).",
    "hierarchical_mapper": "Hierarchical / divide-and-conquer SfM for large "
    "scenes.",
    "pose_prior_mapper": "Incremental SfM seeded and constrained by pose priors "
    "(e.g. GPS).",
    "image_registrator": "Register additional images into an existing model.",
    "point_triangulator": "Triangulate 3D points for a fixed set of known camera "
    "poses.",
    "bundle_adjuster": "Run bundle adjustment to refine an existing model "
    "(Ceres; the production BA).",
    "rig_configurator": "Configure multi-sensor rig constraints in a "
    "database/model.",
    "view_graph_calibrator": "Calibrate camera intrinsics from the view graph.",
    "rotation_averager": "Solve global rotation averaging from relative "
    "rotations.",
    "point_filtering": "Filter 3D points by reprojection error, track length, "
    "and visibility.",
    "color_extractor": "Extract per-point RGB colors from the source images.",
    "image_undistorter": "Undistort images and prepare a dense-stereo workspace "
    "(required before patch_match_stereo).",
    "image_undistorter_standalone": "Undistort images from an external camera/"
    "pose list, without a reconstruction.",
    "image_rectifier": "Stereo-rectify image pairs.",
    "patch_match_stereo": "Dense PatchMatch stereo producing per-image depth and "
    "normal maps. Runs on Metal on Apple Silicon (no CUDA needed); requires "
    "CUDA on other platforms.",
    "stereo_fusion": "Fuse per-image depth/normal maps into a dense 3D point "
    "cloud.",
    "poisson_mesher": "Build a surface mesh from a dense point cloud via Poisson "
    "reconstruction.",
    "delaunay_mesher": "Build a surface mesh via graph-cut Delaunay "
    "tetrahedralization.",
    "advancing_front_mesher": "Build a surface mesh via advancing-front surface "
    "reconstruction.",
    "mesh_simplifier": "Decimate / simplify a triangle mesh.",
    "mesh_texturer": "Texture a mesh using the input images.",
    "model_analyzer": "Print statistics for a model (cameras, images, points, "
    "reprojection error).",
    "model_aligner": "Align a model to a reference (GPS, control points, or "
    "another model).",
    "model_comparer": "Compare two models (pose and point differences).",
    "model_converter": "Convert a model between formats (BIN/TXT/NVM/PLY/...).",
    "model_cropper": "Crop a model to a bounding box or region.",
    "model_clusterer": "Cluster a model into submodels.",
    "model_merger": "Merge two overlapping models into one.",
    "model_splitter": "Split a model into tiles / submodels.",
    "model_transformer": "Apply a similarity or rigid transform to a model.",
    "model_orientation_aligner": "Align a model's orientation to gravity / "
    "coordinate axes.",
    "image_deleter": "Delete images from a model.",
    "image_filterer": "Filter images out of a model by criteria.",
}

# Options shared by (nearly) all pipeline commands: documented once.
COMMON_NAMES = [
    "-h [ --help ]",
    "--project_path",
    "--default_random_seed",
    "--log_target",
    "--log_path",
    "--log_level",
    "--log_severity",
    "--log_color",
    "--progress_format",
    "--progress_every",
    "--quiet",
]


def opt_name(sig):
    """First token identifying an option, e.g. '--log_level' or '-h [ --help ]'."""
    if sig.startswith("-h"):
        return "-h [ --help ]"
    return sig.split()[0]


def md_cell(s):
    """Escape a string for safe use inside a Markdown table cell: neutralize the
    column separator, raw HTML angle brackets, backticks, and embedded newlines."""
    if not s:
        return ""
    return (
        s.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "'")
        .replace("\n", " ")
        .replace("\r", " ")
    )


# Single source of truth for the bundled-model option paths, rendered into both
# llm.txt and README.md so the two cannot drift (each was previously a separate
# hand-maintained list -- and the README copy was missing two entries).
MODEL_PATHS = [
    ("--AlikedExtraction.n16rot_model_path", "models/aliked-n16rot.onnx"),
    ("--AlikedExtraction.n32_model_path", "models/aliked-n32.onnx"),
    ("--AlikedMatching.lightglue_model_path", "models/aliked-lightglue.onnx"),
    (
        "--AlikedMatching.bruteforce_model_path",
        "models/bruteforce-matcher.onnx",
    ),
    ("--SiftMatching.lightglue_model_path", "models/sift-lightglue.onnx"),
    (
        "--VocabTreeMatching.vocab_tree_path",
        "models/vocab_tree_faiss_flickr100K_words256K.bin",
    ),
]
_MP_WIDTH = max(len(f) for f, _ in MODEL_PATHS)


def model_path_lines(indent=""):
    return [f"{indent}{f.ljust(_MP_WIDTH)}  {p}" for f, p in MODEL_PATHS]


GLOG_RE = re.compile(r"^[IWEF]\d{8} ")


def capture_options(cmd):
    """Return [(signature, description)] parsed from `colmap <cmd> -h`."""
    text = run([cmd, "-h"])
    opts = []
    cur = None
    for ln in text.splitlines():
        if not ln.strip() or GLOG_RE.match(ln):
            continue
        if re.match(
            r"^  -", ln
        ):  # option line: exactly two-space indent + dash
            stripped = ln.strip()
            parts = re.split(r"\s{2,}", stripped, maxsplit=1)
            sig = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ""
            cur = [sig, desc]
            opts.append(cur)
        elif cur is not None and re.match(r"^\s{6,}\S", ln):  # continuation
            cur[1] = (cur[1] + " " + ln.strip()).strip()
    return opts


CMD_OPTS = {c: capture_options(c) for c in ALL_CMDS}

# Guard against silently shipping empty docs: every command except these few
# exposes at least the common options, so an empty list means the binary failed
# to launch or its --help format changed. Fail loudly instead. (help/version/gui
# and system_info print custom help, not the standard option list.)
_BENIGN_EMPTY = {"help", "version", "gui", "system_info"}
_empty = [c for c in ALL_CMDS if c not in _BENIGN_EMPTY and not CMD_OPTS[c]]
if _empty:
    raise SystemExit(
        "ERROR: no options parsed for [{}]; refusing to generate docs that would "
        "under-document these commands. Check the binary at {}.".format(
            ", ".join(_empty), BIN
        )
    )

# Canonical usage examples for the headline commands.
EXAMPLES = {
    "automatic_reconstructor": "colmap automatic_reconstructor --image_path IMAGES --workspace_path WORK",
    "feature_extractor": "colmap feature_extractor --image_path IMAGES --database_path DB.db "
    "--FeatureExtraction.use_gpu 0",
    "exhaustive_matcher": "colmap exhaustive_matcher --database_path DB.db --FeatureMatching.use_gpu 0",
    "sequential_matcher": "colmap sequential_matcher --database_path DB.db",
    "vocab_tree_matcher": "colmap vocab_tree_matcher --database_path DB.db "
    "--VocabTreeMatching.vocab_tree_path TREE.bin",
    "mapper": "colmap mapper --image_path IMAGES --database_path DB.db --output_path SPARSE",
    "bundle_adjuster": "colmap bundle_adjuster --input_path SPARSE/0 --output_path SPARSE/0",
    "image_undistorter": "colmap image_undistorter --image_path IMAGES --input_path SPARSE/0 "
    "--output_path DENSE",
    "patch_match_stereo": "colmap patch_match_stereo --workspace_path DENSE",
    "stereo_fusion": "colmap stereo_fusion --workspace_path DENSE --output_path DENSE/fused.ply",
    "poisson_mesher": "colmap poisson_mesher --input_path DENSE/fused.ply "
    "--output_path DENSE/mesh.ply",
    "model_analyzer": "colmap model_analyzer --path SPARSE/0",
    "model_converter": "colmap model_converter --input_path SPARSE/0 --output_path SPARSE/0.ply "
    "--output_type PLY",
    "system_info": "colmap system_info --format json",
}


# =========================================================================
# man pages
# =========================================================================
def troff_escape(s):
    return s.replace("\\", "\\e").replace("-", "\\-")


def write_man(cmd):
    desc = DESC.get(cmd, "COLMAP command.")
    opts = CMD_OPTS.get(cmd, [])
    lines = []
    lines.append(
        '.TH "COLMAP\\-{}" 1 "{}" "{}" "COLMAP Manual"'.format(
            cmd.upper(), DATE, "COLMAP 4.1"
        )
    )
    lines.append(".SH NAME")
    lines.append(f"colmap\\-{cmd} \\- {troff_escape_text(desc)}")
    lines.append(".SH SYNOPSIS")
    lines.append(f".B colmap {cmd}")
    lines.append(".RI [ options ]")
    lines.append(".SH DESCRIPTION")
    lines.append(troff_escape_text(desc))
    lines.append(".PP")
    lines.append(
        "Part of COLMAP. Every option may also be set in an INI "
        "project file passed with \\fB\\-\\-project_path\\fR."
    )
    if opts:
        lines.append(".SH OPTIONS")
        for sig, d in opts:
            lines.append(".TP")
            lines.append(".B " + troff_escape(sig))
            # The signature already shows whether an option takes a value
            # (`arg`) and its default; emit a zero-width body when the binary
            # gives no description, rather than a misleading placeholder.
            lines.append(troff_escape_text(d) if d else "\\&")
    lines.append(".SH SEE ALSO")
    lines.append(".BR colmap (1)")
    lines.append(".SH AUTHOR")
    lines.append("COLMAP contributors. https://colmap.github.io/")
    with open(os.path.join(MAN, f"colmap-{cmd}.1"), "w") as f:
        f.write("\n".join(lines) + "\n")


def troff_escape_text(s):
    # Descriptions: escape backslash and leading control chars only; keep
    # readable. Escape a leading '.' or "'" so troff does not read it as a req.
    s = s.replace("\\", "\\e")
    if s[:1] in (".", "'"):
        s = "\\&" + s
    return s


# Top-level man page: prefer the project's doc/colmap.1.
src_man = os.path.join(REPO, "doc", "colmap.1")
if os.path.exists(src_man):
    with open(src_man) as f:
        top = f.read()
    with open(os.path.join(MAN, "colmap.1"), "w") as f:
        f.write(top)

for c in ALL_CMDS:
    write_man(c)


# =========================================================================
# llm.txt  (dense, complete)
# =========================================================================
def fmt_opt_block(opts, indent="  "):
    out = []
    for sig, d in opts:
        if d:
            out.append(f"{indent}{sig}")
            out.append(f"{indent}    {d}")
        else:
            out.append(f"{indent}{sig}")
    return "\n".join(out)


llm = []
llm.append("# COLMAP CLI — complete reference (LLM-oriented)")
llm.append(VERSION)
llm.append(f"Generated {ISO} from the binary in this folder (./colmap).")
llm.append("")
llm.append("## Build configuration (this binary)")
llm.append("- Platform: macOS, Apple Silicon (arm64).")
llm.append(
    "- GPU compute: Apple Metal — used for dense PatchMatch stereo "
    "(patch_match_stereo). No CUDA."
)
llm.append(
    "- Features: CPU SIFT (VLFeat) and ALIKED (ONNX Runtime). SiftGPU "
    "is unavailable (needs CUDA/OpenGL)."
)
llm.append(
    "- Feature matching runs on CPU by default and is the recommended "
    "path on Apple Silicon (FeatureMatching.use_gpu defaults to 0)."
)
llm.append("- No Qt GUI (the `gui` command is not functional in this build).")
llm.append("")
llm.append("## Invocation")
llm.append("colmap <command> [options]")
llm.append(
    "Any option may instead be set in an INI file passed with "
    "--project_path FILE. Use `colmap <command> -h` for the live option "
    "list. Booleans are 0/1. Unset path options mean 'not provided'."
)
llm.append("")
llm.append("## Typical pipelines")
llm.append("Sparse (SfM):")
llm.append(
    "  colmap feature_extractor --image_path IMAGES --database_path DB.db"
)
llm.append("  colmap exhaustive_matcher --database_path DB.db")
llm.append(
    "  colmap mapper --image_path IMAGES --database_path DB.db "
    "--output_path SPARSE"
)
llm.append("Dense (MVS, Metal on this build):")
llm.append(
    "  colmap image_undistorter --image_path IMAGES --input_path "
    "SPARSE/0 --output_path DENSE"
)
llm.append("  colmap patch_match_stereo --workspace_path DENSE")
llm.append(
    "  colmap stereo_fusion --workspace_path DENSE --output_path "
    "DENSE/fused.ply"
)
llm.append(
    "  colmap poisson_mesher --input_path DENSE/fused.ply --output_path "
    "DENSE/mesh.ply"
)
llm.append(
    "One-shot: colmap automatic_reconstructor --image_path IMAGES "
    "--workspace_path WORK  (sparse only on this build; dense MVS in the "
    "auto pipeline needs CUDA)."
)
llm.append("")
llm.append("## Test data (Hugging Face)")
llm.append("Ready-to-run COLMAP/ETH3D scenes for testing this binary live at "
           "https://huggingface.co/datasets/alexmkwizu/colmap-testing-dataset "
           "(7 scenes, 415 images; each already in COLMAP layout).")
llm.append("Get it with the Hugging Face CLI (pip install huggingface_hub):")
llm.append("  # whole dataset (~3.2 GB)")
llm.append("  hf download alexmkwizu/colmap-testing-dataset --repo-type dataset "
           "--local-dir colmap-testing-dataset")
llm.append("  # one scene only (recommended for a quick test)")
llm.append("  hf download alexmkwizu/colmap-testing-dataset --repo-type dataset "
           "--include 'south-building/**' --local-dir data")
llm.append("Image-path differs by family: COLMAP example scenes "
           "(gerrard-hall, south-building) use <scene>/images; ETH3D scenes "
           "(courtyard, electro, kicker, relief, terrains) use "
           "<scene>/images/dslr_images.")
llm.append("Then run the pipelines above, e.g.:")
llm.append("  colmap feature_extractor --image_path south-building/images "
           "--database_path db.db")
llm.append("  colmap exhaustive_matcher --database_path db.db")
llm.append("  colmap mapper --image_path south-building/images "
           "--database_path db.db --output_path sparse")
llm.append("Each scene ships a ground-truth/reference COLMAP model (ETH3D: "
           "<scene>/dslr_calibration_jpg; COLMAP examples: <scene>/sparse) for "
           "accuracy checks, and the two COLMAP example scenes include a "
           "prebuilt database.db (features+matches) to skip extraction/matching. "
           "See the dataset's own README.md and llm.txt for formats and "
           "per-scene details.")
llm.append("")
llm.append("## Choosing features + matcher (what to use for best output)")
llm.append(
    "Decision drivers: scene difficulty (texture, viewpoint/illumination "
    "change, repetitive structure), image count, and ordered-vs-unordered."
)
llm.append("")
llm.append(
    "FEATURE + MATCHER PAIRINGS (FeatureExtraction.type / FeatureMatching.type):"
)
llm.append(
    "- SIFT + SIFT_BRUTEFORCE: default. Fast, model-free, fully offline. "
    "Best for well-textured, high-overlap, typical captures. Mature and robust."
)
llm.append(
    "- SIFT + SIFT_LIGHTGLUE: SIFT keypoints + LightGlue learned matching. "
    "More robust matching (wide baseline, illumination) with model-free "
    "extraction. Needs sift-lightglue.onnx. Best speed/quality balance."
)
llm.append(
    "- ALIKED_N16ROT + ALIKED_LIGHTGLUE: learned keypoints + learned "
    "matching = highest robustness on HARD scenes (low texture, large "
    "viewpoint/illumination change, repetitive patterns). Costs ALIKED "
    "extraction (~1.4 s/img on CPU) + two models. N16ROT is "
    "rotation-robust (use when images have in-plane rotation)."
)
llm.append(
    "- ALIKED_N32 + ALIKED_LIGHTGLUE: higher-capacity ALIKED variant "
    "(more/denser features) for very hard or high-resolution scenes; "
    "heavier than N16ROT."
)
llm.append(
    "- ALIKED_* + ALIKED_BRUTEFORCE: learned features, cheap cosine "
    "matching. Middle ground; cheaper than LightGlue but weaker matching."
)
llm.append(
    "Tip: extraction and matching feature types MUST agree (SIFT db is "
    "matched by SIFT_*; ALIKED db by ALIKED_*)."
)
llm.append("")
llm.append("MATCHER COMMAND (matching strategy by collection):")
llm.append(
    "- exhaustive_matcher: small/medium UNORDERED sets (up to ~150 imgs). "
    "All pairs; most thorough."
)
llm.append(
    "- sequential_matcher: ORDERED images / video; matches neighbors, "
    "optional vocab-tree loop closure."
)
llm.append("- spatial_matcher: you have GPS / location priors.")
llm.append(
    "- vocab_tree_matcher: LARGE unordered collections (hundreds-"
    "thousands). Needs a vocab tree (bundled: 256K for SIFT; "
    "64K_aliked_n16rot / _n32 for ALIKED)."
)
llm.append("- transitive_matcher: densify an existing match graph.")
llm.append("")
llm.append("MAPPER:")
llm.append("- mapper (incremental): default; robust for small/medium scenes.")
llm.append(
    "- global_mapper: large scenes; rotation averaging + global "
    "positioning, can be faster at scale."
)
llm.append("- hierarchical_mapper: very large scenes (divide & conquer).")
llm.append("")
llm.append("DENSE (MVS) + MESH:")
llm.append(
    "- image_undistorter -> patch_match_stereo (Metal on this build) -> "
    "stereo_fusion -> a mesher."
)
llm.append(
    "- poisson_mesher: smooth watertight surfaces (objects, closed "
    "scenes). delaunay_mesher: detail-preserving, large scenes. "
    "advancing_front_mesher: alternative surface reconstruction."
)
llm.append("")
llm.append("RECOMMENDED RECIPES:")
llm.append(
    "- Best quality on hard scenes: ALIKED_N16ROT + ALIKED_LIGHTGLUE, "
    "exhaustive (small) or vocab_tree (large), incremental mapper, then "
    "Metal dense MVS + poisson/delaunay mesh."
)
llm.append(
    "- Fast / typical well-textured scenes: SIFT + SIFT_BRUTEFORCE "
    "(or sequential for video), incremental mapper."
)
llm.append("- Balanced: SIFT + SIFT_LIGHTGLUE.")
llm.append(
    "- Scaling rule of thumb (CPU LightGlue cost grows with pair count): "
    "prefer the learned matcher for up to ~50-80 images; for larger sets "
    "use SIFT + vocab_tree to keep matching tractable."
)
llm.append("")
llm.append(
    "GPU/CPU ON THIS BUILD: matching runs on CPU (use_gpu 0, faster on "
    "Apple Silicon); dense MVS runs on Metal automatically; ALIKED/"
    "LightGlue ONNX runs on the CPU execution provider."
)
llm.append("")
llm.append("## Offline / bundled models")
llm.append(
    "This build ships with all models alongside the binary; pass explicit "
    "local paths so nothing is downloaded (fully offline):"
)
for _ln in model_path_lines("  "):
    llm.append(_ln)
llm.append(
    "A model-path that is a plain file path is used as-is; the "
    "'<url>;<name>;<sha256>' form would trigger a download (avoid for "
    "offline). SIFT + brute-force and Metal dense MVS need no models."
)
llm.append("")
llm.append("## Common options (accepted by every pipeline command below)")
# Pull the common options' descriptions from a representative command.
rep = CMD_OPTS.get("feature_extractor", [])
common_lookup = {opt_name(s): (s, d) for s, d in rep}
for name in COMMON_NAMES:
    if name in common_lookup:
        s, d = common_lookup[name]
        llm.append("  {}{}".format(s, ("    " + d) if d else ""))
llm.append(
    "Process contract: --progress_format jsonl emits machine-readable "
    "progress events; SIGINT/SIGTERM stop cooperatively, save a partial "
    "result, and exit 130/143."
)
llm.append("")
llm.append("## Commands")
for group, cmds in GROUPS:
    llm.append("")
    llm.append(f"### {group}")
    for c in cmds:
        llm.append("")
        llm.append(f"#### {c}")
        llm.append(DESC.get(c, ""))
        llm.append(f"usage: colmap {c} [options]")
        if c in EXAMPLES:
            llm.append(f"example: {EXAMPLES[c]}")
        opts = CMD_OPTS.get(c, [])
        specific = [
            (s, d) for (s, d) in opts if opt_name(s) not in COMMON_NAMES
        ]
        if c in ("help", "version"):
            llm.append("(no options)")
        elif not opts:
            llm.append(
                "(no options reported; not usable in this build)"
                if c == "gui"
                else "(no command-specific options)"
            )
        else:
            llm.append("options (command-specific; common options also apply):")
            llm.append(
                fmt_opt_block(specific)
                if specific
                else "  (only the common options)"
            )

with open(os.path.join(OUT, "llm.txt"), "w") as f:
    f.write("\n".join(llm) + "\n")

# =========================================================================
# README.md  (human, exhaustive)
# =========================================================================
md = []
md.append("# COLMAP — macOS (Apple Silicon) build")
md.append("")
md.append(f"`{VERSION}`")
md.append("")
md.append(
    "This folder contains a self-contained COLMAP command-line build for "
    "macOS on Apple Silicon, plus complete reference documentation."
)
md.append("")
md.append("| File | What it is |")
md.append("|------|------------|")
md.append("| `colmap` | The CLI binary (stripped, optimized). |")
md.append(
    "| `README.md` | This document — overview, pipelines, and every command. |"
)
md.append("| `llm.txt` | Dense, complete CLI reference for LLM/agent use. |")
md.append("| `man/colmap.1` | Top-level man page. |")
md.append(
    "| `man/colmap-<command>.1` | One man page per subcommand "
    "(%d total). |" % len(ALL_CMDS)
)
md.append("")
md.append(
    "View a man page with: `man -l man/colmap.1` (or "
    "`man -l man/colmap-mapper.1`)."
)
md.append("")
md.append("## This build")
md.append("")
md.append("- **Platform:** macOS, Apple Silicon (arm64).")
md.append(
    "- **GPU compute:** Apple **Metal** — powers dense PatchMatch stereo "
    "(`patch_match_stereo`) with no CUDA required."
)
md.append(
    "- **Features:** CPU SIFT (VLFeat) and ALIKED (ONNX Runtime). "
    "`SiftGPU` is unavailable (it needs CUDA/OpenGL)."
)
md.append(
    "- **Matching:** runs on the **CPU by default** "
    "(`--FeatureMatching.use_gpu 0`), which is the faster and recommended "
    "path on Apple Silicon."
)
md.append(
    "- **No GUI:** built with `GUI_ENABLED=OFF`, so the `gui` command is "
    "not functional here. Everything else is CLI-driven."
)
md.append(
    "- **No CUDA:** CUDA-only paths (SiftGPU, the dense MVS path inside "
    "`automatic_reconstructor`) are unavailable; dense MVS is run "
    "explicitly via `patch_match_stereo` on Metal."
)
md.append("")
md.append("## Running")
md.append("")
md.append("```bash")
md.append("./colmap help                 # list all commands")
md.append("./colmap <command> -h         # options for one command")
md.append("./colmap system_info --format json")
md.append("```")
md.append("")
md.append(
    "If macOS Gatekeeper quarantines the binary, clear it with "
    "`xattr -dr com.apple.quarantine ./colmap`."
)
md.append("")
md.append("## Quick start")
md.append("")
md.append("### Sparse reconstruction (SfM)")
md.append("```bash")
md.append("colmap feature_extractor --image_path IMAGES --database_path DB.db")
md.append("colmap exhaustive_matcher --database_path DB.db")
md.append(
    "colmap mapper --image_path IMAGES --database_path DB.db "
    "--output_path SPARSE"
)
md.append("colmap model_analyzer --path SPARSE/0")
md.append("```")
md.append("")
md.append("### Dense reconstruction (MVS, on Metal)")
md.append("```bash")
md.append(
    "colmap image_undistorter --image_path IMAGES --input_path SPARSE/0 "
    "--output_path DENSE"
)
md.append("colmap patch_match_stereo --workspace_path DENSE")
md.append(
    "colmap stereo_fusion --workspace_path DENSE --output_path DENSE/fused.ply"
)
md.append(
    "colmap poisson_mesher --input_path DENSE/fused.ply --output_path "
    "DENSE/mesh.ply"
)
md.append("```")
md.append("")
md.append(
    "For small/medium unordered photo sets, `exhaustive_matcher` is "
    "best; for ordered/video use `sequential_matcher`; for large "
    "collections use `vocab_tree_matcher`."
)
md.append("")
md.append("## Test data (Hugging Face)")
md.append("")
md.append(
    "Ready-to-run COLMAP + ETH3D scenes for testing this binary are published "
    "at "
    "[`alexmkwizu/colmap-testing-dataset`](https://huggingface.co/datasets/alexmkwizu/colmap-testing-dataset)"
    " — 7 scenes, 415 images, each already in COLMAP's expected layout."
)
md.append("")
md.append("```bash")
md.append("# one scene (quick test); drop --include for the whole ~3.2 GB set")
md.append(
    "hf download alexmkwizu/colmap-testing-dataset --repo-type dataset \\"
)
md.append("  --include 'south-building/**' --local-dir data")
md.append("")
md.append("colmap feature_extractor --image_path data/south-building/images "
          "--database_path db.db")
md.append("colmap exhaustive_matcher --database_path db.db")
md.append("colmap mapper --image_path data/south-building/images "
          "--database_path db.db --output_path sparse")
md.append("```")
md.append("")
md.append(
    "Image paths differ by family: the COLMAP example scenes "
    "(`gerrard-hall`, `south-building`) use `<scene>/images`; the ETH3D scenes "
    "(`courtyard`, `electro`, `kicker`, `relief`, `terrains`) use "
    "`<scene>/images/dslr_images`. Every scene ships a ground-truth/reference "
    "model for accuracy checks (ETH3D: `dslr_calibration_jpg/`; COLMAP: "
    "`sparse/`), and the two COLMAP scenes include a prebuilt `database.db` so "
    "you can skip extraction+matching. The dataset's own `README.md` and "
    "`llm.txt` document the formats and per-scene details."
)
md.append("")
md.append("## Choosing the right pipeline (best output)")
md.append("")
md.append(
    "**Features + matcher** — set `--FeatureExtraction.type` and "
    "`--FeatureMatching.type` (they must agree on the feature family):"
)
md.append("")
md.append("| Pairing | When to use | Cost |")
md.append("|---|---|---|")
md.append(
    "| `SIFT` + `SIFT_BRUTEFORCE` | Default. Well-textured, high-overlap, "
    "typical captures. | Fastest; no models, fully offline |"
)
md.append(
    "| `SIFT` + `SIFT_LIGHTGLUE` | Better matching robustness (wide "
    "baseline, lighting) with model-free extraction. | +`sift-lightglue.onnx` |"
)
md.append(
    "| `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE` | **Best on hard scenes**: low "
    "texture, big viewpoint/illumination change, repetitive structure. "
    "N16ROT handles in-plane rotation. | ALIKED ~1.4 s/img (CPU) + 2 models |"
)
md.append(
    "| `ALIKED_N32` + `ALIKED_LIGHTGLUE` | Higher-capacity ALIKED for very "
    "hard / high-res scenes. | Heaviest |"
)
md.append(
    "| `ALIKED_*` + `ALIKED_BRUTEFORCE` | Learned features, cheap cosine "
    "matching. Middle ground. | Lighter than LightGlue, weaker matching |"
)
md.append("")
md.append(
    "**Matching strategy** — `exhaustive_matcher` (small/medium unordered, "
    "≤~150 imgs) · `sequential_matcher` (ordered/video) · `spatial_matcher` "
    "(GPS priors) · `vocab_tree_matcher` (large unordered; needs a bundled "
    "vocab tree) · `transitive_matcher` (densify graph)."
)
md.append("")
md.append(
    "**Mapper** — `mapper` (incremental, default, small/medium) · "
    "`global_mapper` (large) · `hierarchical_mapper` (very large)."
)
md.append("")
md.append(
    "**Dense + mesh** — `image_undistorter` → `patch_match_stereo` (Metal) "
    "→ `stereo_fusion` → a mesher (`poisson_mesher` for smooth watertight, "
    "`delaunay_mesher` for detail/large, `advancing_front_mesher`)."
)
md.append("")
md.append(
    "**Rule of thumb:** LightGlue matching runs on CPU here, and its cost "
    "grows with the number of pairs — prefer the learned matcher for up to "
    "~50–80 images; for larger sets use `SIFT` + `vocab_tree_matcher` to "
    "keep matching tractable. Keep `--FeatureMatching.use_gpu 0` (CPU is "
    "the path on this build); dense MVS uses Metal automatically."
)
md.append("")
md.append("### Offline / bundled models")
md.append("")
md.append(
    "All models ship next to the binary under `models/`. Pass explicit "
    "local paths so nothing downloads:"
)
md.append("")
md.append("```")
for _ln in model_path_lines():
    md.append(_ln)
md.append("```")
md.append("")
md.append(
    "A plain file path is used as-is; the `<url>;<name>;<sha256>` form "
    "would download (avoid for offline). `SIFT` + brute-force and Metal "
    "dense MVS need no models."
)
md.append("")
md.append("## Common options")
md.append("")
md.append(
    "These are accepted by essentially every pipeline command. Per-"
    "command sections below list only the **command-specific** options; "
    "assume these also apply."
)
md.append("")
md.append("| Option | Meaning |")
md.append("|--------|---------|")
for name in COMMON_NAMES:
    if name in common_lookup:
        s, d = common_lookup[name]
        md.append(f"| `{s}` | {md_cell(d)} |")
md.append("")
md.append(
    "**Process contract.** `--progress_format jsonl` emits machine-"
    "readable per-stage progress events; `--quiet` suppresses non-"
    "essential logging. Long-running commands respond to Ctrl-C "
    "(SIGINT) and SIGTERM by stopping cooperatively, saving a partial "
    "result where possible, and exiting with code 130 / 143."
)
md.append("")
md.append("## Command reference")
md.append("")
for group, cmds in GROUPS:
    md.append(f"### {group}")
    md.append("")
    for c in cmds:
        md.append(f"#### `{c}`")
        md.append("")
        md.append(DESC.get(c, ""))
        md.append("")
        md.append("```")
        md.append(f"colmap {c} [options]")
        if c in EXAMPLES:
            md.append("# example")
            md.append(EXAMPLES[c])
        md.append("```")
        md.append("")
        opts = CMD_OPTS.get(c, [])
        specific = [
            (s, d) for (s, d) in opts if opt_name(s) not in COMMON_NAMES
        ]
        if c in ("help", "version"):
            md.append("*No options.*")
        elif c == "gui":
            md.append("*Not functional in this build (no GUI).*")
        elif not specific:
            md.append("*Only the common options apply.*")
        else:
            md.append(
                "<details><summary>%d command-specific options</summary>"
                % len(specific)
            )
            md.append("")
            md.append("| Option | Description |")
            md.append("|--------|-------------|")
            for s, d in specific:
                md.append(f"| `{s}` | {md_cell(d)} |")
            md.append("")
            md.append("</details>")
        md.append("")

with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write("\n".join(md) + "\n")

print("commands:", len(ALL_CMDS))
print("groups:", [g for g, _ in GROUPS])
print("man pages:", len([f for f in os.listdir(MAN) if f.endswith(".1")]))
print("wrote: README.md, llm.txt, man/")
