ffmpeg -h

ffmpeg version 7.0.1-essentials_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
  built with gcc 13.2.0 (Rev5, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-zlib --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-sdl2 --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxvid --enable-libaom --enable-libopenjpeg --enable-libvpx --enable-mediafoundation --enable-libass --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libgme --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libtheora --enable-libvo-amrwbenc --enable-libgsm --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-librubberband
  libavutil      59.  8.100 / 59.  8.100
  libavcodec     61.  3.100 / 61.  3.100
  libavformat    61.  1.100 / 61.  1.100
  libavdevice    61.  1.100 / 61.  1.100
  libavfilter    10.  1.100 / 10.  1.100
  libswscale      8.  1.100 /  8.  1.100
  libswresample   5.  1.100 /  5.  1.100
  libpostproc    58.  1.100 / 58.  1.100
Universal media converter
usage: ffmpeg [options] [[infile options] -i infile]... {[outfile options] outfile}...

Getting help:
    -h      -- print basic options
    -h long -- print more options
    -h full -- print all options (including all format and codec specific options, very long)
    -h type=name -- print all options for the named decoder/encoder/demuxer/muxer/filter/bsf/protocol
    See man ffmpeg for detailed description of the options.

Per-stream options can be followed by :<stream_spec> to apply that option to specific streams only. <stream_spec> can be a stream index, or v/a/s for video/audio/subtitle (see manual for full syntax).

Print help / information / capabilities:
-L                  show license
-h <topic>          show help
-version            show version
-muxers             show available muxers
-demuxers           show available demuxers
-devices            show available devices
-decoders           show available decoders
-encoders           show available encoders
-filters            show available filters
-pix_fmts           show available pixel formats
-layouts            show standard channel layouts
-sample_fmts        show available audio sample formats

Global options (affect whole program instead of just one file):
-v <loglevel>       set logging level
-y                  overwrite output files
-n                  never overwrite output files
-stats              print progress report during encoding

Per-file options (input and output):
-f <fmt>            force container format (auto-detected otherwise)
-t <duration>       stop transcoding after specified duration
-to <time_stop>     stop transcoding after specified time is reached
-ss <time_off>      start transcoding at specified time


Per-file options (output-only):
-metadata[:<spec>] <key=value>  add metadata

Per-stream options:
-c[:<stream_spec>] <codec>  select encoder/decoder ('copy' to copy stream without reencoding)
-filter[:<stream_spec>] <filter_graph>  apply specified filters to audio/video

Video options:
-r[:<stream_spec>] <rate>  override input framerate/convert to given output framerate (Hz value, fraction or abbreviation)
-aspect[:<stream_spec>] <aspect>  set aspect ratio (4:3, 16:9 or 1.3333, 1.7777)
-vn                 disable video
-vcodec <codec>     alias for -c:v (select encoder/decoder for video streams)
-vf <filter_graph>  alias for -filter:v (apply filters to video streams)
-b <bitrate>        video bitrate (please use -b:v)

Audio options:
-aq <quality>       set audio quality (codec-specific)
-ar[:<stream_spec>] <rate>  set audio sampling rate (in Hz)
-ac[:<stream_spec>] <channels>  set number of audio channels
-an                 disable audio
-acodec <codec>     alias for -c:a (select encoder/decoder for audio streams)
-ab <bitrate>       alias for -b:a (select bitrate for audio streams)
-af <filter_graph>  alias for -filter:a (apply filters to audio streams)

Subtitle options:
-sn                 disable subtitle
-scodec <codec>     alias for -c:s (select encoder/decoder for subtitle streams)
