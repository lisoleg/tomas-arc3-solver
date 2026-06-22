import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Paper, Typography, Box, CircularProgress, Alert, Chip } from '@mui/material';
import { vizApi, SearchTreeData, TreeNode } from '../api/client';

interface KappaSnapTreeProps {
  taskId: string;
}

const KappaSnapTree: React.FC<KappaSnapTreeProps> = ({ taskId }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [data, setData] = useState<SearchTreeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, [taskId]);

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const result = await vizApi.getSearchTree(taskId);
      setData(result);
      drawTree(result.tree);
    } catch (err: any) {
      setError(err.message || 'Failed to load search tree');
    } finally {
      setLoading(false);
    }
  };

  const drawTree = (rootData: TreeNode) => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 500;
    const margin = { top: 30, right: 90, bottom: 30, left: 90 };

    svg.attr('width', width).attr('height', height);

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create hierarchy
    const root = d3.hierarchy(rootData as any);

    // Tree layout
    const treeLayout = d3
      .tree()
      .size([height - margin.top - margin.bottom, width - margin.left - margin.right]);

    treeLayout(root as any);

    // Links
    const linkColor = (d: any) => {
      const phase = d.target.data.phase;
      if (phase === 'phase_a_pass') return '#4caf50';
      if (phase === 'phase_a_fail') return '#f44336';
      return '#999';
    };

    g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal()
        .x((d: any) => d.y)
        .y((d: any) => d.x) as any)
      .attr('fill', 'none')
      .attr('stroke', linkColor)
      .attr('stroke-width', 1.5)
      .attr('opacity', 0.7);

    // Nodes
    const node = g
      .selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d: any) => `translate(${d.y},${d.x})`);

    // Node circles
    node
      .append('circle')
      .attr('r', (d: any) => {
        if (d.data.verified === true) return 8;
        if (d.data.verified === false) return 5;
        return 4;
      })
      .attr('fill', (d: any) => {
        if (d.data.phase === 'phase_a_pass') return '#4caf50';
        if (d.data.phase === 'phase_a_fail') return '#f44336';
        if (d.data.verified === true) return '#2196f3';
        if (d.data.verified === false) return '#ff9800';
        return '#9e9e9e';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5);

    // Node labels
    node
      .append('text')
      .attr('dy', '0.35em')
      .attr('x', (d: any) => (d.children ? -10 : 10))
      .attr('text-anchor', (d: any) => (d.children ? 'end' : 'start'))
      .style('font-size', '10px')
      .style('font-family', 'monospace')
      .text((d: any) => d.data.name);

    // MDL labels on leaf nodes
    node
      .filter((d: any) => !d.children)
      .append('text')
      .attr('dy', '1.2em')
      .attr('x', 10)
      .attr('text-anchor', 'start')
      .style('font-size', '8px')
      .style('fill', '#666')
      .text((d: any) => `MDL:${d.data.mdl}`);

    // Legend
    const legend = svg
      .append('g')
      .attr('transform', 'translate(10, 10)');

    const legendItems = [
      { color: '#4caf50', label: 'Phase A Pass' },
      { color: '#f44336', label: 'Phase A Fail' },
      { color: '#2196f3', label: 'Verified' },
      { color: '#ff9800', label: 'Not Verified' },
    ];

    legendItems.forEach((item, i) => {
      const row = legend
        .append('g')
        .attr('transform', `translate(0, ${i * 18})`);
      row.append('circle').attr('r', 5).attr('fill', item.color);
      row
        .append('text')
        .attr('dy', '0.35em')
        .attr('x', 10)
        .style('font-size', '10px')
        .text(item.label);
    });
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        kappa-Snap Search Tree
      </Typography>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {data && (
        <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip
            label={`Total: ${data.total_candidates}`}
            color="default"
            size="small"
          />
          <Chip
            label={`Phase A Pass: ${data.phase_a_passed}`}
            color="success"
            size="small"
          />
          <Chip
            label={`Phase A Fail: ${data.phase_a_failed}`}
            color="error"
            size="small"
          />
          <Chip
            label={`Verified: ${data.phase_b_verified}`}
            color="info"
            size="small"
          />
          <Chip
            label={`Max Depth: ${data.max_depth}`}
            color="secondary"
            size="small"
          />
        </Box>
      )}

      <Box sx={{ overflowX: 'auto' }}>
        <svg ref={svgRef}></svg>
      </Box>
    </Paper>
  );
};

export default KappaSnapTree;
