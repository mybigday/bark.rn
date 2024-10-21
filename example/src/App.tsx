import { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, View, Text, Pressable } from 'react-native';
import BarkContext from 'bark.rn';

export default function App() {
  const context = useRef<BarkContext | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    BarkContext.load('model.bin').then((ctx) => {
      context.current = ctx;
      setLoaded(true);
    });
  }, []);

  const generate = useCallback(async () => {
    const result = await context.current?.generate(
      'Hello, world!',
      'output.wav'
    );
    console.log(result);
  }, []);

  return (
    <View style={styles.container}>
      <Text>Loaded: {loaded ? 'true' : 'false'}</Text>
      <Pressable onPress={generate}>
        <Text>Generate</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  box: {
    width: 60,
    height: 60,
    marginVertical: 20,
  },
});
