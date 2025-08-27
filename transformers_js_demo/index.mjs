import { pipeline } from '@xenova/transformers';

async function main() {

  console.log("Running Sentiment Analysis...");
  const sentiment = await pipeline('sentiment-analysis');
  const sentimentResult = await sentiment("I love Hugging Face!");
  console.log("Sentiment Result:", sentimentResult);

  console.log("\nRunning Text Generation...");
  const textGen = await pipeline('text-generation');
  const textGenResult = await textGen("Once upon a time, in a world of AI,");
  console.log("Text Generation Result:", textGenResult);

}

main();
