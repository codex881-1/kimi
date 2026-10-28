#!/usr/bin/env bash
# Fetch the OFL fonts used by the film into assets/fonts.
set -euo pipefail
cd "$(dirname "$0")/../assets/fonts"
G=https://github.com/google/fonts/raw/main/ofl
for u in instrumentserif/InstrumentSerif-Italic.ttf instrumentserif/InstrumentSerif-Regular.ttf \
  "inter/Inter%5Bopsz,wght%5D.ttf" "inter/Inter-Italic%5Bopsz,wght%5D.ttf" courierprime/CourierPrime-Regular.ttf \
  courierprime/CourierPrime-Bold.ttf "jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf" ibmplexmono/IBMPlexMono-Regular.ttf \
  ibmplexmono/IBMPlexMono-Medium.ttf ibmplexmono/IBMPlexMono-SemiBold.ttf oldstandardtt/OldStandard-Bold.ttf \
  oldstandardtt/OldStandard-Regular.ttf "notoserifsc/NotoSerifSC%5Bwght%5D.ttf"; do
  f=$(basename "$u" | sed 's/%5B/[/;s/%5D/]/')
  curl -sL -o "$f" "$G/$u"
done
