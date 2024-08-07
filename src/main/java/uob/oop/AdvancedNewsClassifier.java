package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public static List<Glove> getListGlove() {
        return listGlove;
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        List<String> listVocabulary=Toolkit.listVocabulary;
        List<double[]> listVectors=Toolkit.listVectors;
        for (int i = 0; i < listVocabulary.size(); i++) {
            if (!isContains(listVocabulary.get(i), Toolkit.STOPWORDS)) {
                listResult.add(new Glove(listVocabulary.get(i), new Vector(listVectors.get(i))));
            }
        }
        return listResult;
    }

    private static boolean isContains(String word, String[] _stopWords) {
        for (String stopWord : _stopWords) {
            if (word.equals(stopWord)) {
                return true;
            }
        }
        return false;
    }

    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian;
        ArrayList<Integer> list = new ArrayList<>();
        listEmbedding.forEach(articlesEmbedding -> list.add(countValidWords(articlesEmbedding.getNewsContent(), Toolkit.listVocabulary)));
        list.sort(Integer::compareTo);
        int size = list.size();
        if (size % 2 == 0) {
            int average = list.get(size / 2) + list.get((size / 2) + 1);
            intMedian = average / 2;
        } else {
            intMedian = list.get((size + 1) / 2);
        }

        return intMedian;
    }



    private int countValidWords(String _text, List<String> _validWords) {
        String[] words = _text.split("\\s+");
        int count = 0;

        for (String word : words) {
            if (_validWords.contains(word.toLowerCase())) {
                count++;
            }
        }
        return count;
    }
//    for (int i = 0; i < list.size() - 1; i++) {
//        for (int j = 0; j < list.size() - i - 1; j++) {
//            if (list.get(j) > list.get(j + 1)) {
//                int temp = list.get(j);
//                list.set(j, list.get(j + 1));
//                list.set(j + 1, temp);
//            }
//        }

//    }


    public void populateEmbedding() {
        for(ArticlesEmbedding article : listEmbedding){
            try {
                article.getEmbedding();
            } catch (InvalidSizeException f){
                article.setEmbeddingSize(embeddingSize);
            } catch (InvalidTextException g){
                article.getNewsContent();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        List<DataSet> listDS = new ArrayList<>();
        for (ArticlesEmbedding article : listEmbedding) {
            if (article.getNewsType().equals(NewsArticles.DataType.Training)) {
                INDArray inputNDArray = article.getEmbedding();
                INDArray outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                if (article.getNewsLabel().equals("1")) {
                    outputNDArray.putScalar(new int[]{0, 0}, 1);
                } else {
                    outputNDArray.putScalar(new int[]{0, 1}, 1);
                }
                DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                listDS.add(myDataSet);
            }
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }


    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        for (ArticlesEmbedding article : _listEmbedding) {
            if (article.getNewsType().equals(NewsArticles.DataType.Testing)) {
                INDArray input = article.getEmbedding();
                int[] prediction = myNeuralNetwork.predict(input);
                article.setNewsLabel(Integer.toString(prediction[0]));
                listResult.add(prediction[0]);
            }
        }
        return listResult;
    }

    public void printResults() {
        List<String> groupLabels = new ArrayList<>();
        List<List<String>> groups = new ArrayList<>();
        for (ArticlesEmbedding article : listEmbedding) {
            if (article.getNewsType().equals(NewsArticles.DataType.Testing)) {
                String groupLabel = "Group " + (Integer.parseInt(article.getNewsLabel()) + 1);
                int index = groupLabels.indexOf(groupLabel);
                if (index == -1) {
                    groupLabels.add(groupLabel);
                    groups.add(new ArrayList<>());
                    index = groupLabels.size() - 1;
                }
                groups.get(index).add(article.getNewsTitle());
            }
        }

        // Bubble sort the groups based on the group labels
        for (int i = 0; i < groupLabels.size() - 1; i++) {
            for (int j = 0; j < groupLabels.size() - i - 1; j++) {
                if (groupLabels.get(j).compareTo(groupLabels.get(j + 1)) > 0) {
                    // Swap groupLabels[j] and groupLabels[j+1]
                    String tempLabel = groupLabels.get(j);
                    groupLabels.set(j, groupLabels.get(j + 1));
                    groupLabels.set(j + 1, tempLabel);

                    // Swap groups[j] and groups[j+1]
                    List<String> tempGroup = groups.get(j);
                    groups.set(j, groups.get(j + 1));
                    groups.set(j + 1, tempGroup);
                }
            }
        }

        // Print the groups in the sorted order
        for (int i = 0; i < groupLabels.size(); i++) {
            System.out.println(groupLabels.get(i));
            for (String title : groups.get(i)) {
                System.out.println(title);
            }
        }
    }
}
