// screens/growth_chart_screen.dart
// Growth tracking screen using typed GrowthData model.

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../models/growth_data.dart';
import '../services/api_service.dart';
import '../widgets/skill_chip.dart';

class GrowthChartScreen extends StatefulWidget {
  const GrowthChartScreen({super.key});

  @override
  State<GrowthChartScreen> createState() => _GrowthChartScreenState();
}

class _GrowthChartScreenState extends State<GrowthChartScreen> {
  final _api = ApiService();
  bool _loading = true;
  String? _error;
  GrowthData? _data;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() { _loading = true; _error = null; });
    try {
      final data = await _api.growthTracking();
      setState(() => _data = data);
    } catch (e) {
      setState(() => _error = ApiService.extractError(e));
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    if (_loading) return const Center(child: CircularProgressIndicator());

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, size: 48, color: cs.error),
              const SizedBox(height: 12),
              Text(_error!, textAlign: TextAlign.center, style: TextStyle(color: cs.error)),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: _load,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    final data = _data!;

    if (!data.hasData) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.trending_up, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            const Text('No growth data yet', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text(
              data.message,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.grey),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _load,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary stats row 1
            Row(children: [
              _GrowthStat(
                label: 'Total Skills',
                value: '${data.currentSkillCount}',
                delta: data.skillCountDelta,
                icon: Icons.psychology,
              ),
              const SizedBox(width: 12),
              _GrowthStat(
                label: 'Best Match',
                value: data.matchScorePercent,
                deltaLabel: data.scoreDeltaPercent,
                icon: Icons.bar_chart,
                deltaPositive: data.scoreDelta >= 0,
              ),
            ]).animate().fadeIn(),

            const SizedBox(height: 12),

            // Summary stats row 2
            Row(children: [
              _GrowthStat(
                label: 'Analyses',
                value: '${data.totalSnapshots}',
                icon: Icons.history,
              ),
              const SizedBox(width: 12),
              _GrowthStat(
                label: 'New Skills',
                value: '${data.newSkillsCount}',
                icon: Icons.add_circle_outline,
              ),
            ]).animate().fadeIn(delay: 100.ms),

            // Charts (only if 2+ data points)
            if (data.timeSeries.length > 1) ...[
              const SizedBox(height: 28),
              _ChartTitle(title: 'Skill Count Over Time'),
              const SizedBox(height: 12),
              SizedBox(
                height: 200,
                child: LineChart(_buildChart(data.timeSeries, cs, useScore: false)),
              ).animate().fadeIn(delay: 200.ms),

              const SizedBox(height: 28),
              _ChartTitle(title: 'Match Score Over Time'),
              const SizedBox(height: 12),
              SizedBox(
                height: 200,
                child: LineChart(_buildChart(data.timeSeries, cs, useScore: true)),
              ).animate().fadeIn(delay: 300.ms),
            ],

            // Improvement streak badge
            if (data.improvementStreak > 0) ...[
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [cs.primaryContainer, cs.secondaryContainer],
                  ),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Row(
                  children: [
                    const Text('🔥', style: TextStyle(fontSize: 28)),
                    const SizedBox(width: 12),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '${data.improvementStreak}-upload streak!',
                          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                        ),
                        const Text('You\'ve improved with every upload', style: TextStyle(fontSize: 12)),
                      ],
                    ),
                  ],
                ),
              ).animate().fadeIn(delay: 350.ms).scale(begin: const Offset(0.95, 0.95)),
            ],

            // New skills added
            if (data.newSkillsAdded.isNotEmpty) ...[
              const SizedBox(height: 24),
              _ChartTitle(title: 'New Skills Added (${data.newSkillsCount})'),
              const SizedBox(height: 10),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: data.newSkillsAdded
                    .map((s) => SkillChip(skill: s, type: SkillChipType.present))
                    .toList(),
              ).animate().fadeIn(delay: 400.ms),
            ],
          ],
        ),
      ),
    );
  }

  LineChartData _buildChart(
      List<GrowthSnapshot> snapshots,
      ColorScheme cs, {
        required bool useScore,
      }) {
    final spots = snapshots.asMap().entries.map((e) {
      final val = useScore ? e.value.scorePercent : e.value.skillCount.toDouble();
      return FlSpot(e.key.toDouble(), val);
    }).toList();

    return LineChartData(
      gridData: FlGridData(
        show: true,
        drawVerticalLine: false,
        getDrawingHorizontalLine: (_) =>
            FlLine(color: cs.outlineVariant, strokeWidth: 1),
      ),
      titlesData: FlTitlesData(
        leftTitles: AxisTitles(
          sideTitles: SideTitles(showTitles: true, reservedSize: 40),
        ),
        bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
      ),
      borderData: FlBorderData(show: false),
      lineBarsData: [
        LineChartBarData(
          spots: spots,
          isCurved: true,
          color: cs.primary,
          barWidth: 3,
          dotData: FlDotData(show: true),
          belowBarData: BarAreaData(
            show: true,
            color: cs.primary.withOpacity(0.1),
          ),
        ),
      ],
    );
  }
}

class _ChartTitle extends StatelessWidget {
  final String title;
  const _ChartTitle({required this.title});

  @override
  Widget build(BuildContext context) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
    );
  }
}

class _GrowthStat extends StatelessWidget {
  final String label;
  final String value;
  final int? delta;
  final String? deltaLabel;
  final bool? deltaPositive;
  final IconData icon;

  const _GrowthStat({
    required this.label,
    required this.value,
    this.delta,
    this.deltaLabel,
    this.deltaPositive,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isPos = deltaPositive ?? (delta ?? 0) >= 0;
    final deltaStr = deltaLabel ?? (delta != null ? '${isPos ? '+' : ''}$delta' : null);

    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: cs.primary),
            const SizedBox(height: 8),
            Text(value, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            Row(
              children: [
                Flexible(
                  child: Text(label, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
                ),
                if (deltaStr != null) ...[
                  const SizedBox(width: 4),
                  Text(
                    deltaStr,
                    style: TextStyle(
                      color: isPos ? Colors.green : Colors.red,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}