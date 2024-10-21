# bark.rn

Bark.cpp integration for React Native

## Installation

```sh
npm install bark.rn
```

### Setup for iOS with React Native < 0.75

1. Install `cocoapods-spm`

2. Add these lines in `Podfile`

```rb
# ...

target "YourApp" do
  # ...

  spm_pkg "bark",
    :url => "https://github.com/PABannier/bark.cpp.git",
    :branch => "1.0.0",
    :products => ["bark"]

  # ...
end
```

## Usage


```js
import { multiply } from 'bark.rn';

// ...

const result = await multiply(3, 7);
```


## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)