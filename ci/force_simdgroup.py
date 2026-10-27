"""CI-only: the GitHub macOS VM GPU (Apple Paravirtual device) reports MTLGPUFamilyApple5, so ggml-metal turns off
simdgroup reduction / matrix ops and every MUL_MAT falls back to the CPU. GGML_METAL_FORCE_SIMDGROUP=1 turns them on."""
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];\n"
assert s.count(anchor) == 1, "anchor"
s = s.replace(anchor, anchor + """                if (getenv("GGML_METAL_FORCE_SIMDGROUP") != NULL) {
                    dev->props.has_simdgroup_reduction = true;
                    dev->props.has_simdgroup_mm        = true;
                }
""")
open(p, "w").write(s)
print("forced simdgroup override added to", p)
