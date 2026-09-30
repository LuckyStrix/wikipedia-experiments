#!/usr/bin/env bash
# Downloads the enwiki dump files the pipeline uses into dumps/ (or $WIKI_DUMPS).
# Resumable: re-run it and wget -c picks up where it left off. Progress goes to dumps/download.log.
#
# The Kiwix .zim is separate: get wikipedia_en_all_nopic_*.zim from https://download.kiwix.org/zim/wikipedia/
# and put it in dumps/ too.
set -u
DATE=${DUMP_DATE:-20260901}
DUMPS=${WIKI_DUMPS:-"$(dirname "$0")/../dumps"}
mkdir -p "$DUMPS" && cd "$DUMPS"
exec >> download.log 2>&1

base=https://dumps.wikimedia.org/enwiki/$DATE
files=(
  pages-articles-multistream-index.txt.bz2
  page.sql.gz linktarget.sql.gz redirect.sql.gz pagelinks.sql.gz
  categorylinks.sql.gz category.sql.gz
  geo_tags.sql.gz page_props.sql.gz langlinks.sql.gz
  pages-articles-multistream.xml.bz2
)
wget -q -c "$base/enwiki-$DATE-sha1sums.txt"
for f in "${files[@]}"; do
  echo "[$(date '+%F %T')] downloading $f"
  wget -c --progress=dot:giga "$base/enwiki-$DATE-$f" || echo "FAILED: $f"
done
echo "[$(date '+%F %T')] verifying checksums"
sha1sum -c --ignore-missing "enwiki-$DATE-sha1sums.txt"
echo "[$(date '+%F %T')] done"
