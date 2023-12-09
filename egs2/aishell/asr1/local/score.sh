#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <DECODING_DIR>"
  exit 1
fi

script_path="/share/nas165/amian/experiments/speech/SCTK/bin/sclite"

dir=$1 # the path to the decoding dir, e.g. experiments/librispeech_clean100_suffix600_tcpgen500_sche30_nodrop/decode_test_clean_b10_KB1000/
$script_path -r "${dir}/converted.ref.trn" trn -h "${dir}/converted.hyp.trn" trn -i rm -o all stdout > "${dir}/converted.result.wrd.txt"
