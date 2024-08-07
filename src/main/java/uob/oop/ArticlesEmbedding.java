package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";
    private INDArray newsEmbedding = Nd4j.create(0);


    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title, _content, _type, _label);

    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        intSize = _size;
    }

    public int getEmbeddingSize() {
        return intSize;
    }

    //    @Override
//    public String getNewsContent() {
//        //TODO Task 5.3 - 10 Marks
//        StringBuilder finishedText = new StringBuilder();
//        Properties props = new Properties();
//        props.setProperty("annotators", "tokenize,pos,lemma");
//        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//        CoreDocument document = pipeline.processToCoreDocument(textCleaning(super.getNewsContent()));
//
//        for (CoreLabel token : document.tokens()) {
//            String word = token.lemma();
//
//            if (!isStopWord(word)) {
//                finishedText.append(word).append(" ");
//            }
//        }
//        processedText = finishedText.toString().toLowerCase();
//        return processedText.trim();
//    }
//
//    public static boolean isStopWord(String _word) {
//        int low = 0;
//        int high = Toolkit.STOPWORDS.length - 1;
//
//        while (low <= high) {
//            int mid = (low + high) / 2;
//            int comparison = _word.compareTo(Toolkit.STOPWORDS[mid]);
//
//            if (comparison == 0) {
//                return true;
//            } else if (comparison < 0) {
//                high = mid - 1;
//            } else {
//                low = mid + 1;
//            }
//        }
//        return false;
//    }
    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if (this.processedText.equals("")) {
            String cleanedText = textCleaning(super.getNewsContent());
            String lemmatizedText = performLemmatization(cleanedText);
            String processedTextWithoutStopwords = removeStopWords(lemmatizedText);
            this.processedText = processedTextWithoutStopwords.toLowerCase();
        }

        // Return the processed text
        return this.processedText.trim();
    }


    private String performLemmatization(String text) {

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);


        Annotation document = new Annotation(text);


        pipeline.annotate(document);


        StringBuilder lemmatizedText = new StringBuilder();
        for (CoreLabel token : document.get(CoreAnnotations.TokensAnnotation.class)) {
            String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
            lemmatizedText.append(lemma).append(" ");
        }

        return lemmatizedText.toString().trim();
    }

    private String removeStopWords(String text) {
        StringBuilder result = new StringBuilder();

        String[] words = text.split(" ");

        for (String word : words) {
            boolean isStopWord = false;


            for (String stopWord : Toolkit.STOPWORDS) {
                if (word.equals(stopWord)) {
                    isStopWord = true;
                    break;
                }
            }

            if (!isStopWord) {
                result.append(word).append(" ");
            }
        }

        return result.toString().trim();
    }

    public INDArray getEmbedding() throws Exception {
        if (!newsEmbedding.isEmpty()){
            return Nd4j.vstack(newsEmbedding.mean(1));
        }
        if (intSize == -1) {
            throw new InvalidSizeException("Invalid size");
        }
        if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");
        }

        String[] words = processedText.split("\\s+");
        int vectorSize = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize(); // assuming all vectors have the same size
        INDArray embeddingsArray = Nd4j.zeros(intSize, vectorSize);
        int row = 0;
        for (String word : words) {
            if (row >= intSize) {
                break;
            }
            Glove glove = getGloveForWord(word);
            if (glove != null) {
                embeddingsArray.putRow(row++, Nd4j.create(glove.getVector().getAllElements()));
            }
        }

        newsEmbedding = embeddingsArray;
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    private Glove getGloveForWord(String _word) {
        int low = 0;
        int high = Toolkit.listVocabulary.size() - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            String midWord = Toolkit.listVocabulary.get(mid);
            int comparison = _word.compareToIgnoreCase(midWord);

            if (comparison == 0) {
                return new Glove(midWord, new Vector(Toolkit.listVectors.get(mid)));
            } else if (comparison < 0) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return null;
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }

}
