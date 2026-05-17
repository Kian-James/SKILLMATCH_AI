// lib/widgets/progression_chart.dart

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../models/models.dart';
import '../theme.dart';

class ProgressionChart extends StatefulWidget {
  final List<ResumeScanRecord> history;

  const ProgressionChart({super.key, required this.history});

  @override
  State<ProgressionChart> createState() => _ProgressionChartState();
}

class _ProgressionChartState extends State<ProgressionChart> {
  int _touchedIndex = -1;

  /// Deduplicate by filename + target job, keeping the latest scan per combo.
  /// Orders chronologically for the chart.
  List<ResumeScanRecord> get _chartData {
    final seen = <String, ResumeScanRecord>{};
    for (final r in widget.history.reversed) {
      final key = '${r.fileName}__${r.targetJobId}';
      seen[key] = r;
    }
    final sorted = seen.values.toList()
      ..sort((a, b) => a.scannedAt.compareTo(b.scannedAt));
    return sorted;
  }

  @override
  Widget build(BuildContext context) {
    final data = _chartData;
    if (data.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildLegend(),
        const SizedBox(height: 16),
        SizedBox(
          height: 220,
          child: LineChart(
            _buildChart(data),
            duration: const Duration(milliseconds: 500),
            curve: Curves.easeInOut,
          ),
        ),
        if (data.length > 1) ...[
          const SizedBox(height: 12),
          _buildDeltaBanner(data),
        ],
      ],
    );
  }

  Widget _buildLegend() {
    return Row(
      children: [
        _legendDot(AppTheme.teal500, 'Skills'),
        const SizedBox(width: 16),
        _legendDot(AppTheme.amber500, 'Matched skills'),
        const SizedBox(width: 16),
        _legendDot(AppTheme.navy400, 'Certs'),
      ],
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 6),
        Text(label,
            style: GoogleFonts.inter(
                fontSize: 11,
                color: AppTheme.textTertiary,
                fontWeight: FontWeight.w500)),
      ],
    );
  }

  LineChartData _buildChart(List<ResumeScanRecord> data) {
    final skillSpots = <FlSpot>[];
    final matchedSpots = <FlSpot>[];
    final certSpots = <FlSpot>[];

    for (int i = 0; i < data.length; i++) {
      final r = data[i];
      skillSpots.add(FlSpot(i.toDouble(), r.skillCount.toDouble()));
      matchedSpots.add(FlSpot(i.toDouble(), r.matchedCount.toDouble()));
      certSpots.add(FlSpot(i.toDouble(), r.certCount.toDouble()));
    }

    final allY = [
      ...skillSpots,
      ...matchedSpots,
      ...certSpots,
    ].map((s) => s.y).toList();
    final maxY = ((allY.isEmpty ? 10 : allY.reduce((a, b) => a > b ? a : b)) + 3).toDouble();

    return LineChartData(
      minY: 0,
      maxY: maxY,
      gridData: FlGridData(
        show: true,
        drawVerticalLine: false,
        horizontalInterval: (maxY / 4).ceilToDouble().clamp(1, 100),
        getDrawingHorizontalLine: (_) => FlLine(
          color: AppTheme.divider,
          strokeWidth: 1,
        ),
      ),
      borderData: FlBorderData(show: false),
      titlesData: FlTitlesData(
        topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 28,
            interval: (maxY / 4).ceilToDouble().clamp(1, 100),
            getTitlesWidget: (v, _) => Text(
              v.toInt().toString(),
              style: GoogleFonts.inter(
                  fontSize: 10, color: AppTheme.textTertiary),
            ),
          ),
        ),
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 36,
            getTitlesWidget: (v, _) {
              final i = v.toInt();
              if (i < 0 || i >= data.length) return const SizedBox();
              final d = data[i];
              return Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Column(
                  children: [
                    Text(
                      DateFormat('MMM d').format(d.scannedAt),
                      style: GoogleFonts.inter(
                          fontSize: 9, color: AppTheme.textTertiary),
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ),
      lineTouchData: LineTouchData(
        touchCallback: (_, response) {
          setState(() {
            _touchedIndex =
                response?.lineBarSpots?.first.spotIndex ?? -1;
          });
        },
        touchTooltipData: LineTouchTooltipData(
          getTooltipColor: (_) => AppTheme.navy600,
          tooltipRoundedRadius: 8,
          tooltipBorder:
          const BorderSide(color: AppTheme.cardBorder, width: 0.8),
          getTooltipItems: (spots) {
            final i = spots.first.spotIndex;
            if (i < 0 || i >= data.length) return [];
            final r = data[i];
            return [
              LineTooltipItem(
                '${r.skillCount} skills',
                GoogleFonts.inter(
                    fontSize: 12,
                    color: AppTheme.teal500,
                    fontWeight: FontWeight.w600),
              ),
              LineTooltipItem(
                '${r.matchedCount} matched',
                GoogleFonts.inter(fontSize: 11, color: AppTheme.amber500),
              ),
              LineTooltipItem(
                '${r.certCount} certs',
                GoogleFonts.inter(fontSize: 11, color: AppTheme.textTertiary),
              ),
            ];
          },
        ),
      ),
      lineBarsData: [
        // Skills line
        LineChartBarData(
          spots: skillSpots,
          isCurved: true,
          curveSmoothness: 0.35,
          color: AppTheme.teal500,
          barWidth: 2.5,
          isStrokeCapRound: true,
          dotData: FlDotData(
            show: true,
            getDotPainter: (spot, _, __, index) => FlDotCirclePainter(
              radius: index == _touchedIndex ? 6 : 3.5,
              color: AppTheme.teal500,
              strokeWidth: 2,
              strokeColor: AppTheme.navy800,
            ),
          ),
          belowBarData: BarAreaData(
            show: true,
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                AppTheme.teal500.withOpacity(0.18),
                AppTheme.teal500.withOpacity(0.01),
              ],
            ),
          ),
        ),
        // Matched skills line
        LineChartBarData(
          spots: matchedSpots,
          isCurved: true,
          curveSmoothness: 0.35,
          color: AppTheme.amber500,
          barWidth: 2,
          isStrokeCapRound: true,
          dashArray: [6, 4],
          dotData: FlDotData(
            show: true,
            getDotPainter: (_, __, ___, ____) => FlDotCirclePainter(
              radius: 3,
              color: AppTheme.amber500,
              strokeWidth: 1.5,
              strokeColor: AppTheme.navy800,
            ),
          ),
          belowBarData: BarAreaData(show: false),
        ),
        // Certs line
        LineChartBarData(
          spots: certSpots,
          isCurved: true,
          curveSmoothness: 0.35,
          color: AppTheme.navy400,
          barWidth: 1.5,
          isStrokeCapRound: true,
          dotData: FlDotData(
            show: true,
            getDotPainter: (_, __, ___, ____) => FlDotCirclePainter(
              radius: 2.5,
              color: AppTheme.navy400,
              strokeWidth: 1.5,
              strokeColor: AppTheme.navy800,
            ),
          ),
          belowBarData: BarAreaData(show: false),
        ),
      ],
    );
  }

  Widget _buildDeltaBanner(List<ResumeScanRecord> data) {
    final first = data.first;
    final last = data.last;
    final skillDelta = last.skillCount - first.skillCount;
    final matchedDelta = last.matchedCount - first.matchedCount;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppTheme.navy700,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.cardBorder),
      ),
      child: Row(
        children: [
          const Icon(Icons.trending_up_rounded,
              color: AppTheme.teal500, size: 16),
          const SizedBox(width: 8),
          Text(
            'Since first scan: ',
            style: GoogleFonts.inter(
                fontSize: 12, color: AppTheme.textTertiary),
          ),
          _delta('$skillDelta skills', skillDelta, 'skill'),
          const SizedBox(width: 12),
          _delta('$matchedDelta matched', matchedDelta, 'match'),
        ],
      ),
    );
  }

  Widget _delta(String label, int delta, String type) {
    final isPositive = delta >= 0;
    final color = isPositive ? AppTheme.teal400 : AppTheme.red400;
    return Row(
      children: [
        Icon(
          isPositive
              ? Icons.arrow_upward_rounded
              : Icons.arrow_downward_rounded,
          size: 12,
          color: color,
        ),
        const SizedBox(width: 3),
        Text(
          label.replaceAll('-', ''),
          style: GoogleFonts.spaceGrotesk(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: color,
          ),
        ),
      ],
    );
  }
}