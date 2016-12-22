package ml.net;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Softmax1Layer implements Classifier {
	
	public static final double STEP = 1.0;
	private final Random rnd;
	private double[][] weights;
	
	public Softmax1Layer(Random rnd) {
		this.rnd = rnd;
		initWeights();
	}
	
	public static void main(String[] args) {
		Random rnd = new Random();
		List<Sample> trainingSet = new CreateData(rnd).spiral3().getTrain();
		Softmax1Layer softmax1Layer = new Softmax1Layer(rnd);
		Visualise.draw(softmax1Layer, trainingSet);
		softmax1Layer.train(trainingSet);

		Visualise.draw(softmax1Layer, trainingSet);
	}

	@Override
	public int[] classify(List<Sample> samples, boolean dbg) {
		double[][] input = toInputMatrix(samples);
		double[][] act = mm(input, weights);

		int nWrong = 0;
		int[] classes = new int[samples.size()];
		for (int i = 0; i < classes.length; i++) {
			for (int j = 0; j < act[i].length; j++) {
				if (act[i][j] > act[i][classes[i]]) {
					classes[i] = j;
				}
			}
			if (classes[i] != samples.get(i).getLabel()) nWrong++;
		}

		if (dbg) {
			System.out.printf("Classification error: %.2f\n", nWrong / (double) samples.size());
		}

		return classes;
	}

	public void train(List<Sample> samples) {
		double[][] input = toInputMatrix(samples);
		for (int i = 0; i < 100; i++) {
			double[][] act = mm(input, weights);
			double[][] gradient = errorGrad(act, samples);

			double[][] update = mm(-STEP * 1 / (double) samples.size(), mm(transpose(input), gradient));
			weights = add(weights, update);

			classify(samples, true);
		}
	}

	private double[][] add(double[][] a, double[][] b) {
		checkArgument(a.length == b.length && a[0].length == b[0].length);

		double[][] copy = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				copy[i][j] = a[i][j] + b[i][j];
			}
		}
		return copy;
	}

	private double[][] transpose(double[][] m) {
		double[][] copy = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[0].length; j++) {
				copy[j][i] = m[i][j];
			}
		}
		return copy;
	}
	
	private double[][] errorGrad(double[][] act, List<Sample> samples) {
		double[][] copy = new double[act.length][act[0].length];
		for (int i = 0; i < act.length; i++) {
			double sum = 0;
			for (int j = 0; j < act[0].length; j++) {
				copy[i][j] = Math.exp(act[i][j]);
				sum += copy[i][j];
			}
			for (int j = 0; j < act[0].length; j++) {
				copy[i][j] = copy[i][j] / sum;
				if (j == samples.get(i).getLabel()) copy[i][j] -= 1;
			}
		}
		return copy;
	}
	
	private double[][] mm(double a, double[][] b) {
		double[][] copy = new double[b.length][b[0].length];
		for (int i = 0; i < b.length; i++) {
			for (int j = 0; j < b[0].length; j++) {
				copy[i][j] = a * b[i][j];
			}
		}
		return copy;
	}

	private double[][] mm(double[][] a, double[][] b) {
		checkArgument(a[0].length == b.length);

		double[][] copy = new double[a.length][b[0].length];
		for (int i = 0; i < copy.length; i++) {
			for (int j = 0; j < copy[0].length; j++) {
				for (int k = 0; k < a[0].length; k++) {
					copy[i][j] += a[i][k] * b[k][j];
				}
			}
		}

		return copy;
	}

	private double[][] toInputMatrix(List<Sample> samples) {
		double[][] m = new double[samples.size()][samples.get(0).getData().length + 1];

		for (int i = 0; i < samples.size(); i++) {
			double[] data = samples.get(i).getData();

			System.arraycopy(data, 0, m[i], 0, data.length);
			m[i][data.length] = 1;
		}

		return m;
	}

	private void initWeights() {
		weights = new double[2 + 1][3];
		for (int i = 0; i < weights.length; i++) {
			if (i == weights.length - 1) {
				Arrays.fill(weights[i], 0);
			} else {
				for (int j = 0; j < weights[0].length; j++) {
					weights[i][j] = rnd.nextGaussian() * 0.02;
				}
			}
		}
	}
}
