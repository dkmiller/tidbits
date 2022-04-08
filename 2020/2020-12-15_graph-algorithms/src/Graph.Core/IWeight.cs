namespace Graph.Core
{
    interface IWeight<out T>
    {
        double Weight { get; }

        INode<T> Node { get; }
    }
}
