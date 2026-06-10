//! # nexus-viz
//!
//! Dependency-free SVG rendering of SHAP explanation bar charts, replacing the
//! matplotlib output of `get_individual_pred_interpretation`. Produces a
//! horizontal bar chart of the top features, colored by feature group, with a
//! value column and legend. SVG is emitted as a `String` so it can be written
//! to disk, embedded in the desktop app, or returned from the API.

use nexus_core::{CohortAgeStats, FeatureGroup, ShapExplanation};

/// Chart layout configuration.
#[derive(Debug, Clone)]
pub struct ChartStyle {
    pub width: f64,
    pub bar_height: f64,
    pub bar_gap: f64,
    pub margin_left: f64,
    pub margin_right: f64,
    pub margin_top: f64,
    pub margin_bottom: f64,
    pub font_family: String,
}

impl Default for ChartStyle {
    fn default() -> Self {
        Self {
            width: 720.0,
            bar_height: 22.0,
            bar_gap: 8.0,
            margin_left: 260.0,
            margin_right: 40.0,
            margin_top: 70.0,
            margin_bottom: 90.0,
            font_family: "Arial, Helvetica, sans-serif".to_string(),
        }
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Format a feature's value for display, porting `format_value_text`.
fn format_value(
    group: FeatureGroup,
    name: &str,
    value: f64,
    age_stats: Option<CohortAgeStats>,
) -> String {
    match group {
        FeatureGroup::Mutation => format!("{}", value.round() as i64),
        FeatureGroup::Cna => format!("{}", value.round() as i64),
        FeatureGroup::Clinical if name == "Sex" => {
            if value == 1.0 {
                "Male".to_string()
            } else if value == -1.0 {
                "Female".to_string()
            } else {
                "Unknown".to_string()
            }
        }
        FeatureGroup::Clinical if name == "Age" => match age_stats {
            Some(stats) => format!("{}", stats.denormalize(value).round() as i64),
            None => format!("{value:.2}"),
        },
        _ => format!("{value:.2}"),
    }
}

/// Render a SHAP explanation to an SVG document.
///
/// Features are drawn bottom-to-top in ascending SHAP value (matching the
/// Python plot), so the most positive contribution sits at the top.
pub fn render_explanation_svg(
    explanation: &ShapExplanation,
    age_stats: Option<CohortAgeStats>,
    style: &ChartStyle,
) -> String {
    // Draw in ascending SHAP order from the bottom up.
    let mut feats: Vec<_> = explanation.features.iter().collect();
    feats.sort_by(|a, b| a.shap_value.partial_cmp(&b.shap_value).unwrap());

    let n = feats.len().max(1);
    let plot_w = style.width - style.margin_left - style.margin_right;
    let plot_h = n as f64 * (style.bar_height + style.bar_gap);
    let height = style.margin_top + plot_h + style.margin_bottom;

    let max_abs = feats
        .iter()
        .map(|f| f.shap_value.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let zero_x = style.margin_left + plot_w / 2.0;
    let scale = (plot_w / 2.0) / max_abs;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" \
         viewBox=\"0 0 {:.0} {:.0}\" font-family=\"{}\">\n",
        style.width,
        height,
        style.width,
        height,
        xml_escape(&style.font_family)
    ));
    svg.push_str(&format!(
        "<rect width=\"{:.0}\" height=\"{:.0}\" fill=\"white\"/>\n",
        style.width, height
    ));

    // Title block.
    let title = format!(
        "SAMPLE_ID: {} | Prediction: {} (p = {:.3})",
        explanation.sample_id, explanation.predicted_cancer_type, explanation.predicted_probability
    );
    svg.push_str(&format!(
        "<text x=\"{:.0}\" y=\"30\" font-size=\"16\" font-weight=\"bold\">{}</text>\n",
        style.margin_left,
        xml_escape(&title)
    ));
    svg.push_str(&format!(
        "<text x=\"{:.0}\" y=\"52\" font-size=\"12\" fill=\"#555\">SHAP value (impact on model output)</text>\n",
        style.margin_left
    ));

    // Zero reference line.
    svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#bbb\" stroke-dasharray=\"4 3\"/>\n",
        zero_x, style.margin_top, zero_x, style.margin_top + plot_h
    ));

    for (i, f) in feats.iter().enumerate() {
        let y = style.margin_top + i as f64 * (style.bar_height + style.bar_gap);
        let len = f.shap_value.abs() * scale;
        let (x, w) = if f.shap_value >= 0.0 {
            (zero_x, len)
        } else {
            (zero_x - len, len)
        };
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\" opacity=\"0.85\"/>\n",
            x, y, w, style.bar_height, f.group.color()
        ));
        // Feature name + value on the left margin.
        let value_text = format_value(f.group, &f.feature_name, f.feature_value, age_stats);
        let label = format!("{}: {}", f.feature_name, value_text);
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" text-anchor=\"end\">{}</text>\n",
            style.margin_left - 8.0,
            y + style.bar_height * 0.7,
            xml_escape(&label)
        ));
    }

    // Legend.
    let legend_y = style.margin_top + plot_h + 36.0;
    let groups = [
        FeatureGroup::Mutation,
        FeatureGroup::Cna,
        FeatureGroup::Signature,
        FeatureGroup::Clinical,
    ];
    let mut lx = style.margin_left;
    for g in groups {
        svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"14\" height=\"14\" fill=\"{}\"/>\n",
            lx,
            legend_y,
            g.color()
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\">{}</text>\n",
            lx + 20.0,
            legend_y + 12.0,
            xml_escape(g.legend_label())
        ));
        lx += 120.0;
    }

    svg.push_str("</svg>\n");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::ShapFeature;

    fn sample_explanation() -> ShapExplanation {
        ShapExplanation {
            sample_id: "TEST-1".into(),
            predicted_cancer_type: "Non-Small Cell Lung Cancer".into(),
            predicted_probability: 0.85,
            features: vec![
                ShapFeature {
                    feature_name: "ERBB2".into(),
                    group: FeatureGroup::Mutation,
                    shap_value: 0.42,
                    feature_value: 3.0,
                },
                ShapFeature {
                    feature_name: "Age".into(),
                    group: FeatureGroup::Clinical,
                    shap_value: -0.20,
                    feature_value: 0.5,
                },
                ShapFeature {
                    feature_name: "SBS4".into(),
                    group: FeatureGroup::Signature,
                    shap_value: 0.15,
                    feature_value: 0.33,
                },
            ],
        }
    }

    #[test]
    fn renders_valid_svg() {
        let svg = render_explanation_svg(&sample_explanation(), None, &ChartStyle::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("ERBB2"));
        assert!(svg.contains("Non-Small Cell Lung Cancer"));
        // mutation color present
        assert!(svg.contains("red"));
    }

    #[test]
    fn age_denormalized_with_stats() {
        let stats = CohortAgeStats {
            age_mean: 60.0,
            std_mean: 10.0,
        };
        let svg =
            render_explanation_svg(&sample_explanation(), Some(stats), &ChartStyle::default());
        // Age z=0.5 -> 60 + 0.5*10 = 65
        assert!(svg.contains("Age: 65"));
    }
}
