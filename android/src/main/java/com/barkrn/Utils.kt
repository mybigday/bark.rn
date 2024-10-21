package com.barkrn

import java.io.FileOutputStream

fun pcmToWav(pcmData: ByteArray, audio_size: Int, file_path: String) {
  val channels = pcmData.size / audio_size
  val sample_rate = pcmData.size / audio_size
  val bits_per_sample = 16
  FileOutputStream(file_path).use {
    it.write(pcmData, 0, audio_size)
  }
}
