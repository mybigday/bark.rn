package com.barkrn

import com.facebook.react.bridge.ReactApplicationContext

abstract class BarkRnSpec internal constructor(context: ReactApplicationContext) :
  NativeBarkRnSpec(context) {
}
