#!/usr/bin/env bash
# Downloads Wikimedia dump files into dumps/ (or $WIKI_DUMPS), then verifies their checksums.
# Resumable: re-run it and wget -c picks up where it left off. Output also goes to dumps/download.log.
#
#   pipeline/download.sh                       # the default set below
#   pipeline/download.sh page.sql.gz ...       # specific files (names without the enwiki-DATE- prefix)
#
# The Kiwix .zim is separate: see "Getting the data" in the README.
set -u
DATE=${WIKI_DUMP_DATE:-20260901}
DUMPS=${WIKI_DUMPS:-"$(dirname "$0")/../dumps"}
mkdir -p "$DUMPS" && cd "$DUMPS" || exit 1
exec > >(tee -a download.log) 2>&1

if [ $# -gt 0 ]; then
  files=("$@")
else
  files=(
    page.sql.gz redirect.sql.gz linktarget.sql.gz pagelinks.sql.gz
    categorylinks.sql.gz category.sql.gz geo_tags.sql.gz page_props.sql.gz langlinks.sql.gz
    pages-articles-multistream-index.txt.bz2 pages-articles-multistream.xml.bz2
  )
fi
files=("${files[@]/sha1sums.txt}")        # always fetched; drop it from the numbered list
files=($(printf '%s\n' "${files[@]}" | grep .))

base=https://dumps.wikimedia.org/enwiki/$DATE
wget -q -c "$base/enwiki-$DATE-sha1sums.txt" || { echo "can't fetch $base/enwiki-$DATE-sha1sums.txt"; exit 1; }
n=${#files[@]}; i=0; failed=0
for f in "${files[@]}"; do
  i=$((i + 1))
  echo "[$(date '+%F %T')] ($i/$n) downloading $f"
  wget -c --progress=dot:giga "$base/enwiki-$DATE-$f" || { echo "FAILED: $f"; failed=1; }
done
echo "[$(date '+%F %T')] verifying checksums"
names=("${files[@]/#/enwiki-$DATE-}")
grep -F -f <(printf '%s\n' "${names[@]}") "enwiki-$DATE-sha1sums.txt" | sha1sum -c - || failed=1
echo "[$(date '+%F %T')] done"
exit $failed
