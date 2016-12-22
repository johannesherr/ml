package ml.net;

import java.util.List;

public interface Classifier {
	int[] classify(List<Sample> samples, boolean dbg);
}
