package ml.net;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

class CreateData {
	private Random rnd;
	private List<Sample> train;
	private List<Sample> test;

	public CreateData(Random rnd) {
		this.rnd = rnd;
	}

	public List<Sample> getTrain() {
		return train;
	}

	public List<Sample> getTest() {
		return test;
	}

	public CreateData spiral3() {
		int N = 100; // number of points per class
		int D = 2; // dimensionality
		int K = 3; // number of classes
		List<Sample> list = new LinkedList<>();
		for (int j = 0; j < K; j++) {
			for (int i = 0; i < N; i++) {
				double r = 1 / ((double) N) * i;
				double t = ((j + 1) * 4 - j * 4) / ((double) N) * i + (j * 4) + rnd.nextGaussian() * 0.2;

				list.add(new Sample(r * Math.sin(t), r * Math.cos(t), j));
			}
		}
		train = list;
		test = list;
		return this;
	}

	public CreateData spiral() {
		int N = 100; // number of points per class
		int D = 2; // dimensionality
		int K = 2; // number of classes
		List<Sample> list = new LinkedList<>();
		for (int j = 0; j < K; j++) {
			for (int i = 0; i < N; i++) {
				double r = 1 / ((double) N) * i;
				double t = ((j + 1) * 4 - j * 4) / ((double) N) * i + (j * 4) + rnd.nextGaussian() * 0.2;

				list.add(new Sample(r * Math.sin(t), r * Math.cos(t), j));
			}
		}
		train = list;
		test = list;
		return this;
	}

	private int[] zeros(int n) {
		return new int[n];
	}

	private double[][] zeros(int n, int m) {
		return new double[n][m];
	}

	public CreateData invoke2() {
		train = new ArrayList<>();
		test = new ArrayList<>();
		double offset = 2;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 200; j++) {
				if (i == 0) {
					train.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + offset, i));
//						train.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + -offset, i));
				} else {
					train.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + offset, i));
//						train.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + -offset, i));
				}
			}
			for (int j = 0; j < 200; j++) {
				if (i == 0) {
					test.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + offset, i));
					test.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + -offset, i));
				} else {
					test.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + offset, i));
					test.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + -offset, i));
				}
			}
		}
		return this;
	}

	public CreateData invoke1() {
		train = new ArrayList<>();
		test = new ArrayList<>();
		double offset = 2;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 200; j++) {
				if (i == 0) {
					train.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + offset, i));
					train.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + -offset, i));
				} else {
					train.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + offset, i));
					train.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + -offset, i));
				}
			}
			for (int j = 0; j < 200; j++) {
				if (i == 0) {
					test.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + offset, i));
					test.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + -offset, i));
				} else {
					test.add(new Sample(rnd.nextGaussian() + -offset, rnd.nextGaussian() + offset, i));
					test.add(new Sample(rnd.nextGaussian() + offset, rnd.nextGaussian() + -offset, i));
				}
			}
		}
		return this;
	}
}
