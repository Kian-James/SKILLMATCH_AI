// screens/home_screen.dart
// Main shell with bottom navigation.
// Shows banner for anonymous users encouraging account creation.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../services/auth_service.dart';
import 'upload_screen.dart';
import 'recommendations_screen.dart';
import 'skill_gap_screen.dart';
import 'growth_chart_screen.dart';
import 'auth/register_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  final List<Widget> _screens = const [
    UploadScreen(),
    RecommendationsScreen(),
    SkillGapScreen(),
    GrowthChartScreen(),
  ];

  final List<NavigationDestination> _destinations = const [
    NavigationDestination(
      icon: Icon(Icons.upload_file_outlined),
      selectedIcon: Icon(Icons.upload_file),
      label: 'Resume',
    ),
    NavigationDestination(
      icon: Icon(Icons.work_outline),
      selectedIcon: Icon(Icons.work),
      label: 'Jobs',
    ),
    NavigationDestination(
      icon: Icon(Icons.track_changes_outlined),
      selectedIcon: Icon(Icons.track_changes),
      label: 'Skill Gap',
    ),
    NavigationDestination(
      icon: Icon(Icons.trending_up_outlined),
      selectedIcon: Icon(Icons.trending_up),
      label: 'Growth',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthService>();
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(_appBarTitle(_selectedIndex)),
        actions: [
          // User avatar / account button
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: IconButton(
              onPressed: () => _showAccountSheet(context),
              icon: CircleAvatar(
                radius: 18,
                backgroundColor: cs.primaryContainer,
                child: Text(
                  _userInitial(auth),
                  style: TextStyle(
                    color: cs.primary,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // Anonymous user upgrade banner
          if (auth.isAnonymous) _buildAnonBanner(context),
          Expanded(child: _screens[_selectedIndex]),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: _destinations,
      ).animate().slideY(begin: 1, duration: 400.ms, curve: Curves.easeOut),
    );
  }

  String _appBarTitle(int index) {
    const titles = ['Upload Resume', 'Job Matches', 'Skill Gap', 'Growth'];
    return titles[index];
  }

  String _userInitial(AuthService auth) {
    if (auth.isAnonymous) return '?';
    final name = auth.currentUser?.displayName ?? auth.currentUser?.email ?? '';
    return name.isNotEmpty ? name[0].toUpperCase() : 'U';
  }

  Widget _buildAnonBanner(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return MaterialBanner(
      backgroundColor: cs.tertiaryContainer,
      content: Text(
        'You\'re browsing as a guest. Create an account to save your progress.',
        style: TextStyle(color: cs.onTertiaryContainer, fontSize: 13),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.push(
            context,
            MaterialPageRoute(builder: (_) => const RegisterScreen()),
          ),
          child: const Text('Create Account'),
        ),
      ],
    );
  }

  void _showAccountSheet(BuildContext context) {
    final auth = context.read<AuthService>();
    final user = auth.currentUser;

    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.person_outline),
              title: Text(
                auth.isAnonymous
                    ? 'Guest User'
                    : (user?.displayName ?? 'User'),
              ),
              subtitle: Text(
                auth.isAnonymous
                    ? 'Not signed in permanently'
                    : (user?.email ?? ''),
              ),
            ),
            if (auth.isAnonymous)
              ListTile(
                leading: const Icon(Icons.person_add_outlined),
                title: const Text('Create Account'),
                subtitle: const Text('Save progress permanently'),
                onTap: () {
                  Navigator.pop(ctx);
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const RegisterScreen()),
                  );
                },
              ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.logout, color: Colors.red),
              title: const Text('Sign Out', style: TextStyle(color: Colors.red)),
              onTap: () async {
                Navigator.pop(ctx);
                await auth.signOut();
              },
            ),
          ],
        ),
      ),
    );
  }
}
