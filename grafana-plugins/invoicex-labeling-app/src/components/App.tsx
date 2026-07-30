import React, { Suspense } from 'react';
import { createBrowserRouter, Navigate, RouterProvider, useRouteError } from 'react-router-dom';
import { AppRootProps } from '@grafana/data';

const QueuePage = React.lazy(() => import('../pages/QueuePage').then((m) => ({ default: m.QueuePage })));
const LabelPage = React.lazy(() => import('../pages/LabelPage').then((m) => ({ default: m.LabelPage })));
const SchemaEditorPage = React.lazy(() =>
  import('../pages/SchemaEditorPage').then((m) => ({ default: m.SchemaEditorPage }))
);
const ModelPage = React.lazy(() => import('../pages/ModelPage').then((m) => ({ default: m.ModelPage })));
const ResultsPage = React.lazy(() => import('../pages/ResultsPage').then((m) => ({ default: m.ResultsPage })));

function LazyRoute({ component: Component }: { component: React.ComponentType }) {
  return <Component />;
}

function RouteErrorBoundary() {
  const error = useRouteError();
  const message = error instanceof Error ? error.message : 'Unexpected plugin route error';

  return <div>{message}</div>;
}

export function App(props: AppRootProps) {
  const router = React.useMemo(
    () =>
      createBrowserRouter(
        [
          { path: '/queue', element: <LazyRoute component={QueuePage} />, errorElement: <RouteErrorBoundary /> },
          { path: '/label/:documentId', element: <LazyRoute component={LabelPage} />, errorElement: <RouteErrorBoundary /> },
          { path: '/label', element: <LazyRoute component={LabelPage} />, errorElement: <RouteErrorBoundary /> },
          { path: '/schema', element: <LazyRoute component={SchemaEditorPage} />, errorElement: <RouteErrorBoundary /> },
          { path: '/model', element: <LazyRoute component={ModelPage} />, errorElement: <RouteErrorBoundary /> },
          { path: '/results', element: <LazyRoute component={ResultsPage} />, errorElement: <RouteErrorBoundary /> },
          { path: '*', element: <Navigate to="/queue" replace />, errorElement: <RouteErrorBoundary /> },
        ],
        { basename: props.basename || '/a/invoicex-labeling-app' }
      ),
    [props.basename]
  );

  return (
    <Suspense fallback={<div />}>
      <RouterProvider router={router} />
    </Suspense>
  );
}
