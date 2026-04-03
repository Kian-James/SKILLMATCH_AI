// screens/home_screen.dart
// Main shell with 2-tab bottom nav: Home and Growth.
// Jobs and Skill Gap are accessed via buttons on the Home tab,
// only visible after a resume has been uploaded/scanned.

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';

import '../services/auth_service.dart';
import 'upload_screen.dart';
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
    GrowthChartScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthService>();
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('SkillMatch AI'),
        actions: [
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
          if (auth.isAnonymous) _buildAnonBanner(context),
          Expanded(child: _screens[_selectedIndex]),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.trending_up_outlined),
            selectedIcon: Icon(Icons.trending_up),
            label: 'Growth',
          ),
        ],
      ).animate().slideY(begin: 1, duration: 400.ms, curve: Curves.easeOut),
    );
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
              width: 40, height: 4,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.person_outline),
              title: Text(auth.isAnonymous ? 'Guest User' : (user?.displayName ?? 'User')),
              subtitle: Text(auth.isAnonymous ? 'Not signed in permanently' : (user?.email ?? '')),
            ),
            if (auth.isAnonymous)
              ListTile(
                leading: const Icon(Icons.person_add_outlined),
                title: const Text('Create Account'),
                subtitle: const Text('Save progress permanently'),
                onTap: () {
                  Navigator.pop(ctx);
                  Navigator.push(context,
                      MaterialPageRoute(builder: (_) => const RegisterScreen()));
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