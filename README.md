# bark.rn

Bark.cpp integration for React Native

## Installation

```sh
npm install bark.rn
```

## Usage


```js
import BarkContext from 'bark.rn';

// Load model
const ctx = await BarkContext.load('path/to/model.bin');

// Inference
const result = await ctx.generate('Hello, world!', 'path/to/output.wav');
// Result: { success: boolean, load_time: number, eval_time: number }

// Release context
await ctx.release();
```


## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
