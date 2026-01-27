import 'package:flutter/material.dart';
import '../services/skill_gap_service.dart';
import '../models/skill_gap_model.dart';
import '../widgets/skill_chip.dart';

class SkillGapScreen extends StatefulWidget {
  final String userId;

  const SkillGapScreen({Key? key, required this.userId}) : super(key: key);

  @override
  _SkillGapScreenState createState() => _SkillGapScreenState();
}

class _SkillGapScreenState extends State<SkillGapScreen> {
  bool _isLoading = false;
  SkillGapModel? _skillGap;
  String _error = '';
  final TextEditingController _roleController = TextEditingController();

  Future<void> _analyzeSkillGap() async {
    final role = _roleController.text.trim();
    if (role.isEmpty) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Please enter a job role')));
      return;
    }

    setState(() {
      _isLoading = true;
      _error = '';
      _skillGap = null;
    });

    try {
      final result = await SkillGapService.analyzeSkillGap(widget.userId, role);
      setState(() {
        _skillGap = result;
      });
    } catch (e) {
      setState(() {
        _error = 'Error analyzing skill gap: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildSkillList(String title, List<String> skills) {
    if (skills.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        const SizedBox(height: 6),
        Wrap(
          spacing: 6,
          runSpacing: 4,
          children: skills.map((s) => SkillChip(skill: s)).toList(),
        ),
        const SizedBox(height: 12),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Skill Gap Analysis')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Job role input
            TextField(
              controller: _roleController,
              decoration: const InputDecoration(
                labelText: 'Enter Job Role',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),

            ElevatedButton(
              onPressed: _analyzeSkillGap,
              child: const Text('Analyze Skill Gap'),
            ),

            const SizedBox(height: 20),

            // Display loading or results
            if (_isLoading)
              const Center(child: CircularProgressIndicator())
            else if (_error.isNotEmpty)
              Center(child: Text(_error))
            else if (_skillGap != null)
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildSkillList('Present Skills', _skillGap!.presentSkills),
                        _buildSkillList('Missing Skills', _skillGap!.missingSkills),
                        _buildSkillList('Suggested Certifications', _skillGap!.suggestedCertifications),
                      ],
                    ),
                  ),
                ),
          ],
        ),
      ),
    );
  }
}
