// screens/growth_chart_screen.dart
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';

class GrowthChartScreen extends StatefulWidget {
  const GrowthChartScreen({super.key});
  @override
  State<GrowthChartScreen> createState() => _GrowthChartScreenState();
}

class _GrowthChartScreenState extends State<GrowthChartScreen> {
  final _api = ApiService();
  bool _loading = true;
  String? _error;
  Map<String, dynamic>? _data;

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
    if (_loading) return const Center(child: CircularProgressIndicator());
    if (_error != null) return Center(child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)));
    if (_data == null || _data!['has_data'] == false) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.trending_up, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            const Text('No growth data yet', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text(_data?['message'] ?? 'Upload your resume to start tracking.',
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.grey)),
          ],
        ),
      );
    }

    final ts = List<Map<String, dynamic>>.from(_data!['time_series'] ?? []);
    final cs = Theme.of(context).colorScheme;

    return RefreshIndicator(
      onRefresh: _load,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary stats
            Row(children: [
              _GrowthStat(label: 'Total Skills', value: '${_data!['current_skill_count']}',
                delta: _data!['skill_count_delta'], icon: Icons.psychology),
              const SizedBox(width: 12),
              _GrowthStat(label: 'Best Match', value: '${((_data!['current_best_match_score'] as num) * 100).toStringAsFixed(0)}%',
                delta: ((_data!['score_delta'] as num) * 100).round(), icon: Icons.bar_chart),
            ]).animate().fadeIn(),

            const SizedBox(height: 12),

            Row(children: [
              _GrowthStat(label: 'Analyses', value: '${_data!['total_snapshots']}', icon: Icons.history),
              const SizedBox(width: 12),
              _GrowthStat(label: 'New Skills', value: '${_data!['new_skills_count']}', icon: Icons.add_circle_outline),
            ]).animate().fadeIn(delay: 100.ms),

            const SizedBox(height: 24),

            // Skill count over time chart
            if (ts.length > 1) ...[
              Text('Skill Count Over Time', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
              const SizedBox(height: 12),
              SizedBox(
                height: 200,
                child: LineChart(_buildLineChart(ts, cs, useScore: false)),
              ).animate().fadeIn(delay: 200.ms),
              const SizedBox(height: 24),

              Text('Match Score Over Time', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
              const SizedBox(height: 12),
              SizedBox(
                height: 200,
                child: LineChart(_buildLineChart(ts, cs, useScore: true)),
              ).animate().fadeIn(delay: 300.ms),
            ],

            // New skills added
            if ((_data!['new_skills_added'] as List).isNotEmpty) ...[
              const SizedBox(height: 24),
              Text('New Skills Added', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
              const SizedBox(height: 10),
              Wrap(
                spacing: 8, runSpacing: 8,
                children: List<String>.from(_data!['new_skills_added'])
                  .map((s) => Chip(
                    label: Text(s, style: TextStyle(fontSize: 12, color: cs.onSecondaryContainer)),
                    backgroundColor: cs.secondaryContainer,
                    side: BorderSide.none,
                  )).toList(),
              ).animate().fadeIn(delay: 400.ms),
            ],
          ],
        ),
      ),
    );
  }

  LineChartData _buildLineChart(List<Map<String, dynamic>> ts, ColorScheme cs, {required bool useScore}) {
    final spots = ts.asMap().entries.map((e) {
      final val = useScore
        ? ((e.value['best_match_score'] as num).toDouble() * 100)
        : (e.value['skill_count'] as num).toDouble();
      return FlSpot(e.key.toDouble(), val);
    }).toList();

    return LineChartData(
      gridData: FlGridData(show: true, drawVerticalLine: false,
        getDrawingHorizontalLine: (_) => FlLine(color: cs.outlineVariant, strokeWidth: 1)),
      titlesData: FlTitlesData(
        leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 40)),
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

class _GrowthStat extends StatelessWidget {
  final String label;
  final String value;
  final int? delta;
  final IconData icon;
  const _GrowthStat({required this.label, required this.value, this.delta, required this.icon});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isPositive = (delta ?? 0) >= 0;
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(14)),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, size: 20, color: cs.primary),
            const SizedBox(height: 8),
            Text(value, style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            Row(children: [
              Text(label, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
              if (delta != null) ...[
                const SizedBox(width: 4),
                Text(
                  '${isPositive ? '+' : ''}$delta',
                  style: TextStyle(
                    color: isPositive ? Colors.green : Colors.red,
                    fontSize: 11, fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ]),
          ],
        ),
      ),
    );
  }
}
