package ml.net;

import java.awt.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import javax.swing.*;

public class Visualise {

	public static void draw(final Classifier classifier, List<Sample> train) {
		ArrayList<Sample> copy = new ArrayList<>(train);
		int[] classified = classifier.classify(copy, true);

		List<Sample> bg = new LinkedList<>();
		double res = 0.05;
		for (double x = -3; x < 3; x += res) {
			for (double y = -3; y < 3; y += res) {
				bg.add(new Sample(x, y, 0));
			}
		}
		int[] bgClasses = classifier.classify(bg, false);

		int w = 800;
		int h = 600;
		JFrame jFrame = new JFrame();
		jFrame.setContentPane(new JPanel() {
			@Override
			public Dimension getPreferredSize() {
				return new Dimension(w, h);
			}

			@Override
			protected void paintComponent(Graphics g) {
				super.paintComponent(g);

				Color[] colors = new Color[]{Color.red, Color.blue, Color.green};
				Color[] colorsBG = new Color[]{Color.decode("#ffbbbb"), Color.decode("#bbbbff"), Color.decode("#bbffbb")};
				int[] diff = {w / 2, h / 2};
				double scale = h / 5.0;

				for (int i = 0; i < bg.size(); i++) {
					Sample sample = bg.get(i);
					g.setColor(colorsBG[bgClasses[i]]);
					int x = translateX(diff, scale, sample.getData()[0] - res / 2);
					int y = translateY(diff, scale, sample.getData()[1] - res / 2);
					g.fillRect(x, y, (int) (res * scale), (int) (res * scale));
				}

				for (int i = 0; i < copy.size(); i++) {
					Sample sample = copy.get(i);
					g.setColor(colors[sample.getLabel()]);
					double[] data = sample.getData();
					int x = translateX(diff, scale, data[0]);
					int y = translateY(diff, scale, data[1]);

					int s = 10;
					if (classified[i] == 0) {
						g.drawRect(x, y, s, s);
					} else if (classified[i] == 1) {
						g.drawChars(new char[]{'X'}, 0, 1, x, y);
					} else {
						g.drawArc(x, y, s, s, 0, 360);
					}

//					g.setColor(Color.black);
//					for (double[] sep : softmax1.weights1) {
//						int x1 = (int) (scale * -2 + diff[0]);
//						int y1 = (int) (scale * -(-sep[0] * -2 - sep[2]) / sep[1] + diff[1]);
//						int x2 = (int) (scale * 2 + diff[0]);
//						int y2 = (int) (scale * -(-sep[0] * 2 - sep[2]) / sep[1] + diff[1]);
//						g.drawLine(x1, y1, x2, y2);
//					}
				}

			}
		});
		jFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		jFrame.pack();
		jFrame.setVisible(true);

//		pp(softmax.weights1);
	}

	private static int translateY(int[] diff, double scale, double y) {
		return (int) (scale * -y + diff[1]);
	}

	private static int translateX(int[] diff, double scale, double x) {
		return (int) (scale * x + diff[0]);
	}
}
