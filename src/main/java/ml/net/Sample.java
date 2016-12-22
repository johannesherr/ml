package ml.net;

public class Sample {
	private final double[] data;
	private final int label;
	public Sample(double x, double y, int label) {
		this.data = new double[]{x, y};
		this.label = label;
	}

	public double[] getData() {
		return data;
	}

	public int getLabel() {
		return label;
	}
}
